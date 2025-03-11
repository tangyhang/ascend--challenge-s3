#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <typename T>
class KernelMatMul1
{
public:
    __aicore__ inline KernelMatMul1() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, const TCubeTiling &tiling, AscendC::TPipe *pipe)
    {
        this->tiling = tiling;
        xGm1.SetGlobalBuffer((__gm__ T *)x1, tiling.M * tiling.Ka);
        xGm2.SetGlobalBuffer((__gm__ T *)x2, tiling.Kb * tiling.N);
        xGm3.SetGlobalBuffer((__gm__ T *)x3, tiling.M * tiling.N);
        yGm.SetGlobalBuffer((__gm__ T *)y, tiling.M * tiling.N);
        int offset1 = 0, offset2 = 0, offset3 = 0;
        CalcOffset(AscendC::GetBlockIdx(), tiling, offset1, offset2, offset3);
        xGm1 = xGm1[offset1];
        xGm2 = xGm2[offset2];
        xGm3 = xGm3[offset3];
        yGm = yGm[offset3];
        pipe->InitBuffer(outQueue, 1, tiling.baseM * tiling.baseN * sizeof(T));
        pipe->InitBuffer(biasQueue, 1, tiling.baseM * tiling.baseN * sizeof(T));
    }
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offset1, int32_t &offset2, int32_t &offset3)
    {
        auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
        auto mCoreIndx = blockIdx % mSingleBlocks;
        auto nCoreIndx = blockIdx / mSingleBlocks;
        offset1 = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        offset2 = nCoreIndx * tiling.singleCoreN;
        offset3 = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
        this->singleCoreM = tiling.singleCoreM;
        this->singleCoreN = tiling.singleCoreN;
        int tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
        tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
        int tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
        tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
        if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN)
        {
            matmulObj.SetTail(tailM, tailN);
            this->singleCoreM = tailM;
            this->singleCoreN = tailN;
        }
    }
    __aicore__ inline void Process()
    {
        uint32_t count = 0;
        const uint32_t roundM = (singleCoreM + tiling.baseM - 1) / tiling.baseM;
        const uint32_t roundN = (singleCoreN + tiling.baseN - 1) / tiling.baseN;
        uint16_t tailM = singleCoreM % tiling.baseM == 0 ? tiling.baseM : singleCoreM % tiling.baseM;
        uint16_t tailN = singleCoreN % tiling.baseN == 0 ? tiling.baseN : singleCoreN % tiling.baseN;
        matmulObj.SetTensorA(xGm1);
        matmulObj.SetTensorB(xGm2);
        while (matmulObj.Iterate())
        {
            out_local = outQueue.AllocTensor<T>();
            matmulObj.GetTensorC(out_local, false, true);
            bias_local = biasQueue.AllocTensor<T>();
            uint16_t curM = count % roundM == roundM - 1 ? tailM : tiling.baseM;
            uint16_t curN = count / roundM == roundN - 1 ? tailN : tiling.baseN;
            uint32_t startOffset = (count % roundM * tiling.baseM * tiling.N + count / roundM * tiling.baseN);
            DataCopyParams copyParam1 = {(uint16_t)curM, (uint16_t)(curN * sizeof(T) / 32), (uint16_t)((tiling.N - curN) * sizeof(T) / 32), 0};
            DataCopy(bias_local, xGm3[startOffset], copyParam1);
            biasQueue.EnQue(bias_local);
            biasQueue.DeQue<T>();
            bias_local.GetValue(0);
            Sub(out_local, out_local, bias_local, tiling.baseM * tiling.baseN);
            biasQueue.FreeTensor(bias_local);
            outQueue.EnQue(out_local);
            outQueue.DeQue<T>();
            DataCopyParams copyParam2 = {(uint16_t)curM, (uint16_t)(curN * sizeof(T) / 32), 0, (uint16_t)((tiling.N - curN) * sizeof(T) / 32)};
            DataCopy(yGm[startOffset], out_local, copyParam2);
            outQueue.FreeTensor(out_local);
            count ++;
        }
        matmulObj.End();
    }
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, T>>
        matmulObj;

private:
    AscendC::GlobalTensor<T> xGm1;
    AscendC::GlobalTensor<T> xGm2;
    AscendC::GlobalTensor<T> xGm3;
    AscendC::GlobalTensor<T> yGm;
    AscendC::LocalTensor<T> out_local;
    AscendC::LocalTensor<T> bias_local;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> outQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> biasQueue;
    int32_t singleCoreM, singleCoreN;
    TCubeTiling tiling;
};

template <typename T>
class KernelMatMulSub2
{
public:
    __aicore__ inline KernelMatMulSub2() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, const TCubeTiling &tiling, AscendC::TPipe *pipe)
    {
        this->tiling = tiling;
        xGm1.SetGlobalBuffer((__gm__ T *)x1, tiling.M * tiling.Ka);
        xGm2.SetGlobalBuffer((__gm__ T *)x2, tiling.Kb * tiling.N);
        xGm3.SetGlobalBuffer((__gm__ T *)x3, tiling.M * tiling.N);
        yGm.SetGlobalBuffer((__gm__ T *)y, tiling.M * tiling.N);
        int offset1 = 0, offset2 = 0, offset3 = 0;
        CalcOffset(AscendC::GetBlockIdx(), tiling, offset1, offset2, offset3);
        xGm1 = xGm1[offset1];
        xGm2 = xGm2[offset2];
        xGm3 = xGm3[offset3];
        yGm = yGm[offset3];
        pipe->InitBuffer(outQueue, 1, tiling.singleCoreN * sizeof(T));
        pipe->InitBuffer(biasQueue, 1, tiling.singleCoreN * sizeof(T));
    }
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offset1, int32_t &offset2, int32_t &offset3)
    {
        auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
        auto mCoreIndx = blockIdx % mSingleBlocks;
        auto nCoreIndx = blockIdx / mSingleBlocks;
        offset1 = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        offset2 = nCoreIndx * tiling.singleCoreN;
        offset3 = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
        this->singleCoreM = tiling.singleCoreM;
        this->singleCoreN = tiling.singleCoreN;
        int tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
        tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
        int tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
        tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
        if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN)
        {
            matmulObj.SetTail(tailM, tailN);
            this->singleCoreM = tailM;
            this->singleCoreN = tailN;
        }
    }
    __aicore__ inline void Process()
    {
        uint32_t count = 0;
        matmulObj.SetTensorA(xGm1);
        matmulObj.SetTensorB(xGm2);
        matmulObj.IterateAll(yGm);
        matmulObj.End();
        out_local = outQueue.AllocTensor<T>();
        bias_local = biasQueue.AllocTensor<T>();
        for(int i = 0; i < singleCoreM; i ++)
        {
            DataCopyExtParams copyParam = {1, (uint32_t)(singleCoreN * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParam{false, 0, 0, 0};
            DataCopyPad(out_local, yGm[i * tiling.N], copyParam, padParam);
            DataCopyPad(bias_local, xGm3[i * tiling.N], copyParam, padParam);
            outQueue.EnQue(out_local);
            biasQueue.EnQue(bias_local);
            outQueue.DeQue<T>();
            biasQueue.DeQue<T>();
            out_local.GetValue(0);
            bias_local.GetValue(0);
            Sub(out_local, out_local, bias_local, singleCoreN);
            outQueue.EnQue(out_local);
            outQueue.DeQue<T>();
            DataCopyExtParams copyParam1 = {1, (uint32_t)(singleCoreN * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm[i * tiling.N], out_local, copyParam1);
            out_local.GetValue(0);
        }
        outQueue.FreeTensor(out_local);
        biasQueue.FreeTensor(bias_local);
    }
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
        matmulObj;

private:
    AscendC::GlobalTensor<T> xGm1;
    AscendC::GlobalTensor<T> xGm2;
    AscendC::GlobalTensor<T> xGm3;
    AscendC::GlobalTensor<T> yGm;
    AscendC::LocalTensor<T> out_local;
    AscendC::LocalTensor<T> bias_local;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> outQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> biasQueue;
    int32_t singleCoreM, singleCoreN;
    TCubeTiling tiling;
};

template <typename T>
class KernelMatMulSub3
{
public:
    __aicore__ inline KernelMatMulSub3() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, const TCubeTiling &tiling, AscendC::TPipe *pipe)
    {
        this->tiling = tiling;
        xGm1.SetGlobalBuffer((__gm__ T *)x1, tiling.M * tiling.Ka);
        xGm2.SetGlobalBuffer((__gm__ T *)x2, tiling.Kb * tiling.N);
        xGm3.SetGlobalBuffer((__gm__ T *)x3, tiling.N);
        yGm.SetGlobalBuffer((__gm__ T *)y, tiling.M * tiling.N);
        int offset1 = 0, offset2 = 0, offset3 = 0, offsety = 0;
        CalcOffset(AscendC::GetBlockIdx(), tiling, offset1, offset2, offset3, offsety);
        xGm1 = xGm1[offset1];
        xGm2 = xGm2[offset2];
        xGm3 = xGm3[offset3];
        yGm = yGm[offsety];
        pipe->InitBuffer(biasQueue, 1, tiling.singleCoreN * sizeof(T));
    }
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offset1, int32_t &offset2, int32_t &offset3, int32_t &offsety)
    {
        auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
        auto mCoreIndx = blockIdx % mSingleBlocks;
        auto nCoreIndx = blockIdx / mSingleBlocks;
        offset1 = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        offset2 = nCoreIndx * tiling.singleCoreN;
        offset3 = nCoreIndx * tiling.singleCoreN;
        offsety = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
        int tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
        tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
        int tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
        tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
        if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN)
            matmulObj.SetTail(tailM, tailN);
    }
    __aicore__ inline void Process()
    {
        bias_local = biasQueue.AllocTensor<T>();
        uint32_t copysz = (tiling.singleCoreN * sizeof(T) + 31) / 32 * 32 / sizeof(T);
        DataCopy(bias_local, xGm3, copysz);
        biasQueue.EnQue(bias_local);
        biasQueue.DeQue<T>();
        Muls(bias_local, bias_local, (T)-1.0, (int32_t)tiling.singleCoreN);
        matmulObj.SetTensorA(xGm1);
        matmulObj.SetTensorB(xGm2);
        matmulObj.SetBias(bias_local);
        matmulObj.IterateAll(yGm);
        matmulObj.End();
        biasQueue.FreeTensor(bias_local);
    }
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
        matmulObj;

private:
    AscendC::GlobalTensor<T> xGm1;
    AscendC::GlobalTensor<T> xGm2;
    AscendC::GlobalTensor<T> xGm3;
    AscendC::GlobalTensor<T> yGm;
    AscendC::LocalTensor<T> bias_local;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> biasQueue;
    TCubeTiling tiling;
};

template <typename T>
class KernelMatMulSub4
{
public:
    __aicore__ inline KernelMatMulSub4() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, const TCubeTiling &tiling, AscendC::TPipe *pipe)
    {
        this->tiling = tiling;
        xGm1.SetGlobalBuffer((__gm__ T *)x1, tiling.M * tiling.Ka);
        xGm2.SetGlobalBuffer((__gm__ T *)x2, tiling.Kb * tiling.N);
        xGm3.SetGlobalBuffer((__gm__ T *)x3, tiling.M * tiling.N);
        yGm.SetGlobalBuffer((__gm__ T *)y, tiling.M * tiling.N);
        int offset1 = 0, offset2 = 0, offset3 = 0;
        CalcOffset(AscendC::GetBlockIdx(), tiling, offset1, offset2, offset3);
        xGm1 = xGm1[offset1];
        xGm2 = xGm2[offset2];
        xGm3 = xGm3[offset2];
        yGm = yGm[offset3];
        pipe->InitBuffer(outQueue, 1, tiling.singleCoreN * sizeof(T));
        pipe->InitBuffer(biasQueue, 1, tiling.singleCoreN * sizeof(T));
    }
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offset1, int32_t &offset2, int32_t &offset3)
    {
        auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
        auto mCoreIndx = blockIdx % mSingleBlocks;
        auto nCoreIndx = blockIdx / mSingleBlocks;
        offset1 = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        offset2 = nCoreIndx * tiling.singleCoreN;
        offset3 = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
        this->singleCoreM = tiling.singleCoreM;
        this->singleCoreN = tiling.singleCoreN;
        int tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
        tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
        int tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
        tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
        if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN)
        {
            matmulObj.SetTail(tailM, tailN);
            this->singleCoreM = tailM;
            this->singleCoreN = tailN;
        }
    }
    __aicore__ inline void Process()
    {
        uint32_t count = 0;
        matmulObj.SetTensorA(xGm1);
        matmulObj.SetTensorB(xGm2);
        matmulObj.IterateAll(yGm);
        matmulObj.End();
        out_local = outQueue.AllocTensor<T>();
        bias_local = biasQueue.AllocTensor<T>();
        for(int i = 0; i < singleCoreM; i ++)
        {
            DataCopyExtParams copyParam = {1, (uint32_t)(singleCoreN * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParam{false, 0, 0, 0};
            DataCopyPad(out_local, yGm[i * tiling.N], copyParam, padParam);
            DataCopyPad(bias_local, xGm3, copyParam, padParam);
            outQueue.EnQue(out_local);
            biasQueue.EnQue(bias_local);
            outQueue.DeQue<T>();
            biasQueue.DeQue<T>();
            out_local.GetValue(0);
            bias_local.GetValue(0);
            Sub(out_local, out_local, bias_local, singleCoreN);
            outQueue.EnQue(out_local);
            outQueue.DeQue<T>();
            DataCopyExtParams copyParam1 = {1, (uint32_t)(singleCoreN * sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm[i * tiling.N], out_local, copyParam1);
            out_local.GetValue(0);
        }
        outQueue.FreeTensor(out_local);
        biasQueue.FreeTensor(bias_local);
    }
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
        matmulObj;

private:
    AscendC::GlobalTensor<T> xGm1;
    AscendC::GlobalTensor<T> xGm2;
    AscendC::GlobalTensor<T> xGm3;
    AscendC::GlobalTensor<T> yGm;
    AscendC::LocalTensor<T> out_local;
    AscendC::LocalTensor<T> bias_local;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> outQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> biasQueue;
    int32_t singleCoreM, singleCoreN;
    TCubeTiling tiling;
};

extern "C" __global__ __aicore__ void mat_mul_sub(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if (TILING_KEY_IS(1))
    {
        KernelMatMul1<DTYPE_X1> op;
        AscendC::TPipe pipe;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tiling_data.cubeTilingData);
        op.Init(x1, x2, x3, y, tiling_data.cubeTilingData, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(2))
    {
        KernelMatMulSub2<DTYPE_X1> op;
        AscendC::TPipe pipe;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tiling_data.cubeTilingData);
        op.Init(x1, x2, x3, y, tiling_data.cubeTilingData, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(3))
    {
        KernelMatMulSub3<DTYPE_X1> op;
        AscendC::TPipe pipe;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tiling_data.cubeTilingData);
        op.Init(x1, x2, x3, y, tiling_data.cubeTilingData, &pipe);
        op.Process();
    }
    else if(TILING_KEY_IS(4))
    {
        KernelMatMulSub4<DTYPE_X1> op;
        AscendC::TPipe pipe;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tiling_data.cubeTilingData);
        op.Init(x1, x2, x3, y, tiling_data.cubeTilingData, &pipe);
        op.Process();
    }
}