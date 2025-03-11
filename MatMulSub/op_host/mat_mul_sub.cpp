
#include "mat_mul_sub_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <vector>
#define MATMUL_TILING_0 1
#define MATMUL_TILING_1 2
#define MATMUL_TILING_2 3
#define MATMUL_TILING_3 4
using namespace matmul_tiling;

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        MatMulSubTilingData tiling;
        auto dt = context->GetInputTensor(0)->GetDataType();
        const gert::StorageShape *x1_shape = context->GetInputShape(0);
        const gert::StorageShape *x2_shape = context->GetInputShape(1);
        const gert::StorageShape *x3_shape = context->GetInputShape(2);
        int32_t M = x1_shape->GetStorageShape().GetDim(0);
        int32_t N = x2_shape->GetStorageShape().GetDim(1);
        int32_t K = x1_shape->GetStorageShape().GetDim(1);
        int32_t biasDim = x3_shape->GetStorageShape().GetDimNum();
        int32_t baseM = 1, baseN = 1, sz;
        while (baseM <= M / 2 && baseM < 128)
            baseM *= 2;
        while (baseN <= N / 4 && baseN < 128)
            baseN *= 2;
        if(baseM < 16)
            baseM = 16;
        if(baseN < 16)
            baseN = 16;
        int count = (M + baseM - 1) / baseM * (N + baseN - 1) / baseN;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
        if(count >= 40)
        {
            cubeTiling.SetDim(40);
        }
        else
            cubeTiling.SetDim(2);
        if (dt == ge::DT_FLOAT)
        {
            sz = 8;
            cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
            cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
            if(N % sz == 0)
                cubeTiling.SetCType(TPosition::VECIN, CubeFormat::ND, DataType::DT_FLOAT);
            else
                cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
            if (biasDim == 1)
                cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        }
        else
        {
            sz = 16;
            cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
            cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
            if(N % sz == 0)
                cubeTiling.SetCType(TPosition::VECIN, CubeFormat::ND, DataType::DT_FLOAT16);
            else
                cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
            if (biasDim == 1)
                cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        }
        cubeTiling.SetShape(M, N, K);
        cubeTiling.SetOrgShape(M, N, K);
        cubeTiling.SetFixSplit(baseM, baseN, -1);
        if (biasDim == 2)
        {
            cubeTiling.SetBias(false);
            if(N % sz == 0)
                context->SetTilingKey(MATMUL_TILING_0);
            else
                context->SetTilingKey(MATMUL_TILING_1);
            }
        else
        {
            cubeTiling.SetBias(true);
            if (dt == ge::DT_FLOAT)
                context->SetTilingKey(MATMUL_TILING_2);
            else
                context->SetTilingKey(MATMUL_TILING_3);
        }
        cubeTiling.SetBufferSpace(-1, -1, -1);
        if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1)
        {
            return ge::GRAPH_FAILED;
        }
        if(count >= 40 && biasDim == 1 && dt == ge::DT_FLOAT)
            context->SetBlockDim(20);
        else
            context->SetBlockDim(1);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t userWorkspaceSize = 0;
        size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
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
        *y_shape = *x1_shape;
        y_shape->SetDim(1, x2_shape->GetDim(1));
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class MatMulSub : public OpDef
    {
    public:
        explicit MatMulSub(const char *name) : OpDef(name)
        {
            this->Input("x1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("x2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("x3")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(MatMulSub);
}
