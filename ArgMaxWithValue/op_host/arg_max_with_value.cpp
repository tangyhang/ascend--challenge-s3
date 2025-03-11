
#include "arg_max_with_value_tiling.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"

#define REDUCE_TILING_0 0
#define REDUCE_TILING_1 1

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ArgMaxWithValueTilingData tiling;
    //从attr获取dimension属性值
    // const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const uint32_t* reduceDim = context->GetAttrs()->GetAttrPointer<uint32_t>(0);
    //获取reduceDim轴的长度
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    // const gert::Shape& xShape = x1_shape->GetStorageShape();
    const uint32_t reduceAxisLen = x1_shape->GetStorageShape().GetDim(*reduceDim);
    tiling.set_reduceAxisLen(reduceAxisLen);
    
    //获取迭代次数(输出的长度)
    // const uint32_t totalLength = x1_shape->GetOriginShape().GetShapeSize();
    const uint32_t outputLength = x1_shape->GetOriginShape().GetShapeSize() / reduceAxisLen;
    tiling.set_outputLength(outputLength);
    
    uint32_t useCoreNum = (outputLength < 40) ? outputLength : 40;
    uint32_t smallCoreProcessNum = outputLength / useCoreNum;
    uint32_t bigCoreNum = outputLength % useCoreNum;
    uint32_t bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
    tiling.set_bigCoreNum(bigCoreNum);
    tiling.set_bigCoreProcessNum(bigCoreProcessNum);
    tiling.set_smallCoreProcessNum(smallCoreProcessNum);

    if (context->GetInputTensor(0)->GetDataType() == ge::DT_FLOAT)
    {
        context->SetTilingKey(REDUCE_TILING_0);
    }
    else
    {
        //获取子张量的长度
        int32_t subTensorLen = 1;
        for (int i = *reduceDim + 1; i < x1_shape->GetStorageShape().GetDimNum(); i++)
            subTensorLen *= x1_shape->GetStorageShape().GetDim(i);
        tiling.set_subTensorLen(subTensorLen);
        useCoreNum = 1;
        context->SetTilingKey(REDUCE_TILING_1);
    }

    context->SetBlockDim(useCoreNum);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // const gert::Shape* x_shape = context->GetInputShape(0);
    
    // // 获取输入张量的维度
    // int dim_count = x_shape->GetDimNum();
    
    // // 读取输入属性，获取"dimension"参数
    // const gert::RuntimeAttrs* attrs = context->GetAttrs();
    // const uint32_t* reduceDim = attrs->GetAttrPointer<uint32_t>(0);
    // const uint32_t dimension = *reduceDim;

    // // 计算输出形状
    // gert::Shape* indice_shape = context->GetOutputShape(0);
    // gert::Shape* values_shape = context->GetOutputShape(1);

    // // 设置输出形状：indice和values的形状与输入相同，但在"dimension"轴上维度被压缩
    // *indice_shape = *x_shape;
    // *values_shape = *x_shape;
    // if (dim_count > 0) {
    //     // 对指定的"dimension"轴进行压缩
    //     indice_shape->SetDim(dimension, 1);
    //     values_shape->SetDim(dimension, 1);
    // }

    return ge::GRAPH_SUCCESS;
}


static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class ArgMaxWithValue : public OpDef {
public:
    explicit ArgMaxWithValue(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("indice")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dimension").Int();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(ArgMaxWithValue);
}
