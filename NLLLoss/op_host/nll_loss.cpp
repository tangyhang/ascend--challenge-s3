
#include "nll_loss_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#define NLLL_TILING_0 1
#define NLLL_TILING_1 2
#define NLLL_TILING_2 3
const char *MODE[3] = {"mean", "sum", "none"};

namespace optiling
{
    const uint32_t BUFFER_NUM = 2;
    const uint32_t BLOCK_SIZE = 32;
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        NLLLossTilingData tiling;
        // auto ret = context->SetNeedAtomic(true);
        const gert::StorageShape *x1_shape = context->GetInputShape(0);
        if (x1_shape->GetStorageShape().GetDimNum() == 1)
        {
            tiling.set_tensorCount(x1_shape->GetStorageShape().GetDim(0));
            tiling.set_blockLen(0);
            const gert::RuntimeAttrs *attrs = context->GetAttrs();
            const char *reduction = attrs->GetStr(0);
            if (!strcmp(reduction, MODE[0]))
                context->SetTilingKey(NLLL_TILING_0);
            else if (!strcmp(reduction, MODE[1]))
                context->SetTilingKey(NLLL_TILING_1);
            else
                context->SetTilingKey(NLLL_TILING_2);
            context->SetBlockDim(1);
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        }
        else
        {
            uint32_t tensorCount = x1_shape->GetStorageShape().GetDim(0);
            uint32_t blockLen = x1_shape->GetStorageShape().GetDim(1);
            tiling.set_tensorCount(tensorCount);
            tiling.set_blockLen(blockLen);
            const gert::RuntimeAttrs *attrs = context->GetAttrs();
            const char *reduction = attrs->GetStr(0);
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
            auto coreNum = ascendcPlatform.GetCoreNum();
            uint32_t useCoreNum = (tensorCount < coreNum) ? tensorCount : coreNum;
            uint32_t smallCoreProcessNum = tensorCount / useCoreNum;
            uint32_t bigCoreNum = tensorCount % useCoreNum;
            uint32_t bigCoreProcessNum = (bigCoreNum == 0) ? smallCoreProcessNum : smallCoreProcessNum + 1;
            tiling.set_bigCoreNum(bigCoreNum);
            tiling.set_bigCoreProcessNum(bigCoreProcessNum);
            tiling.set_smallCoreProcessNum(smallCoreProcessNum);
            if (!strcmp(reduction, MODE[0]))
            {
                context->SetTilingKey(NLLL_TILING_0);
                context->SetBlockDim(1);
            }
            else if (!strcmp(reduction, MODE[1]))
            {
                context->SetTilingKey(NLLL_TILING_1);
                context->SetBlockDim(useCoreNum);
            }
            else
            {
                context->SetTilingKey(NLLL_TILING_2);
                context->SetBlockDim(1);
            }
            tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
            context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        }
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        const gert::Shape *x2_shape = context->GetInputShape(1);
        gert::Shape *y_shape = context->GetOutputShape(0);
        if (x1_shape->GetDimNum() == 1)
        {
            y_shape->SetDimNum(1);
            y_shape->SetDim(0, 1);
        }
        else
        {
            *y_shape = *x2_shape;
        }
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class NLLLoss : public OpDef
    {
    public:
        explicit NLLLoss(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("target")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("weight")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
                // .InitValue(0);
            this->Attr("reduction").AttrType(OPTIONAL).String("mean");
            this->Attr("ignore_index").AttrType(OPTIONAL).Int(-100);

            this->SetInferShape(ge::InferShape);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(NLLLoss);
}
