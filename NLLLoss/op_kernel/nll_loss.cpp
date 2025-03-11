#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include <limits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

class KernelNLLLossMean
{
public:
    __aicore__ inline KernelNLLLossMean() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, uint32_t tensorCount,
                                uint32_t blockLen)
    {
        this->tensorCount = tensorCount;            // N
        this->classCount = blockLen; // C
        this->blockLen = blockLen;                  // C * d1 * .. * dk
        if (blockLen == 0)
        {
            xGm.SetGlobalBuffer((__gm__ float *)x, tensorCount);
            targetGm.SetGlobalBuffer((__gm__ int32_t *)target, tensorCount);
            weightGm.SetGlobalBuffer((__gm__ float *)weight, tensorCount);
            yGm.SetGlobalBuffer((__gm__ float *)y, 1);
        }
        else
        {
            xGm.SetGlobalBuffer((__gm__ float *)x, tensorCount * blockLen);
            targetGm.SetGlobalBuffer((__gm__ int32_t *)target, tensorCount);
            weightGm.SetGlobalBuffer((__gm__ float *)weight, this->classCount);
            yGm.SetGlobalBuffer((__gm__ float *)y, 1);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLen == 0)
        {
            int target = targetGm.GetValue(0);
            float x = xGm.GetValue(target);
            float y = -1 * x;
            yGm.SetValue(0, y);
        }
        else
        {
            float sum_x = 0, sum_w = 0;
            for (int i = 0; i < tensorCount; i++)
            {
                int k = targetGm.GetValue(i);
                float w = weightGm.GetValue(k);
                float x = xGm.GetValue(i * blockLen + k);
                sum_x += x * w;
                sum_w += w;
            }
            yGm.SetValue(0, -1 * sum_x / sum_w);
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<int32_t> targetGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> yGm;
    uint32_t tensorCount;
    uint32_t classCount;
    uint32_t blockLen;
    uint32_t processNum;
};

class KernelNLLLossSum
{
public:
    __aicore__ inline KernelNLLLossSum() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, uint32_t tensorCount,
                                uint32_t blockLen, uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreProcessNum, TPipe *pipe)
    {
        this->tensorCount = tensorCount; // N
        this->blockLen = blockLen;         // C * d1 * .. * dk
        if (blockLen == 0)
        {
            xGm.SetGlobalBuffer((__gm__ float *)x, tensorCount);
            targetGm.SetGlobalBuffer((__gm__ int32_t *)target, tensorCount);
            weightGm.SetGlobalBuffer((__gm__ float *)weight, tensorCount);
            yGm.SetGlobalBuffer((__gm__ float *)y, 1);
        }
        else
        {
            this->blockIdx = GetBlockIdx();
            uint32_t globalBufferIndex = bigCoreProcessNum * blockIdx;
            if (blockIdx < bigCoreNum)
                this->processNum = bigCoreProcessNum;
            else
            {
                this->processNum = smallCoreProcessNum;
                globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (blockIdx - bigCoreNum);
            }
            xGm.SetGlobalBuffer((__gm__ float *)x + globalBufferIndex * this->blockLen, this->blockLen * this->processNum);
            targetGm.SetGlobalBuffer((__gm__ int32_t *)target + globalBufferIndex, this->processNum);
            weightGm.SetGlobalBuffer((__gm__ float *)weight, this->blockLen);
            yGm.SetGlobalBuffer((__gm__ float *)y, 1);
            // if (blockIdx == 0)
            // {
            //     InitGlobalMemory(yGm, 1, (float)0);
            // }
            // pipe->InitBuffer(outQueueY, BUFFER_NUM, 32);
            // pipe->InitBuffer(inQueueX, this->processNum, 32);
            // pipe->InitBuffer(inQueueT, BUFFER_NUM, this->processNum * 4);
            // pipe->InitBuffer(inQueueW, BUFFER_NUM, this->blockLen * 4);

            pipe->InitBuffer(outQueueY, BUFFER_NUM, this->processNum * 4);
            pipe->InitBuffer(inQueueT, BUFFER_NUM, this->processNum * 4);
            pipe->InitBuffer(inQueueW, BUFFER_NUM, this->blockLen * 4);
            pipe->InitBuffer(inQueueX, BUFFER_NUM, this->blockLen * this->processNum * 4);
            pipe->InitBuffer(offQueue, BUFFER_NUM, this->processNum * 4);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLen == 0)
        {
            int target = targetGm.GetValue(0);
            float x = xGm.GetValue(target);
            float w = weightGm.GetValue(target);
            float y = -1 * x * w;
            yGm.SetValue(0, y);
        }
        else
        {
            if(blockIdx == 0)
            {
                yGm.SetValue(0, (float)0);
                DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(yGm);
            }
            CopyIn();
            Compute();
            CopyOut();
            
            // float sum = 0;
            // for (int32_t i = 0; i < processNum; i++)
            // {
            //     int k = targetGm.GetValue(i);
            //     float w = weightGm.GetValue(k);
            //     float x = xGm.GetValue(i * blockLen + k);
            //     sum -= w * x;
            // }
            // LocalTensor<float> y_local = outQueueY.AllocTensor<float>();
            // y_local.SetValue(0, sum);
            // DataCopyExtParams copyParams = {1, (uint32_t)(sizeof(float)), 0, 0, 0};
            // SetAtomicAdd<float>();
            // DataCopyPad(yGm, y_local, copyParams);

            
            // float sum = 0;
            // LocalTensor<int32_t> t_local = inQueueT.AllocTensor<int32_t>();
            // LocalTensor<float> w_local = inQueueW.AllocTensor<float>();

            // DataCopy(t_local, targetGm, (this->processNum + 7)  / 8 * 8);
            // DataCopy(w_local, weightGm, (this->blockLen + 7)  / 8 * 8);
            // inQueueT.EnQue(t_local);
            // inQueueT.DeQue<int32_t>();
            // inQueueW.EnQue(w_local);
            // inQueueW.DeQue<float>();

            // for (int32_t i = 0; i < processNum; i++)
            // {
            //     LocalTensor<float> x_local = inQueueX.AllocTensor<float>();
            //     int k = t_local.GetValue(i);
            //     DataCopy(x_local, xGm[i * this->blockLen + k], 8);
            //     float w = w_local.GetValue(k);
            //     inQueueX.EnQue(x_local);
            //     inQueueX.DeQue<float>();
            //     // PipeBarrier<PIPE_MTE2>();
            //     float x = x_local.GetValue(0);
            //     sum -= w * x;
            // }
            // LocalTensor<float> y_local = outQueueY.AllocTensor<float>();
            // y_local.SetValue(0, sum);
            // DataCopyExtParams copyParams = {1, (uint32_t)(sizeof(float)), 0, 0, 0};
            // SetAtomicAdd<float>();
            // DataCopyPad(yGm, y_local, copyParams);
        }
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<int32_t> t_local = inQueueT.AllocTensor<int32_t>();
        DataCopy(t_local, targetGm, (this->processNum + 7)  / 8 * 8);
        inQueueT.EnQue(t_local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<int32_t> t_local = inQueueT.DeQue<int32_t>();
        LocalTensor<int32_t> off_local = offQueue.AllocTensor<int32_t>();
        LocalTensor<float> y_local = outQueueY.AllocTensor<float>();

        LocalTensor<float> w_local = inQueueW.AllocTensor<float>();
        DataCopy(w_local, weightGm, (this->blockLen + 7)  / 8 * 8);
        LocalTensor<float> x_local = inQueueX.AllocTensor<float>();
        DataCopy(x_local, xGm, (this->blockLen * this->processNum + 7) / 8 * 8);
        if(processNum <= 64)
        {
            ArithProgression<int32_t>(off_local, 0, (int32_t)(blockLen * 4), processNum);
            SetMaskCount();
            SetVectorMask<int32_t, AscendC::MaskMode::COUNTER>(processNum);
            Muls<int32_t, false>(t_local, t_local, (int32_t)4, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 8, 8 });
            Add<int32_t, false>(off_local, off_local, t_local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
            inQueueW.EnQue(w_local);
            inQueueW.DeQue<float>();
            SetMaskNorm();
            Gather(w_local, w_local, t_local.ReinterpretCast<uint32_t>(), (uint32_t)0, processNum);
            inQueueX.EnQue(x_local);
            inQueueX.DeQue<float>();
            Gather(x_local, x_local, off_local.ReinterpretCast<uint32_t>(), (uint32_t)0, processNum);
            Mul(x_local, x_local, w_local, processNum);
            ReduceSum<float>(y_local, x_local, x_local, processNum);
        }
        else
        {
            ArithProgression<int32_t>(off_local, 0, (int32_t)(blockLen * 4), processNum);
            Muls(t_local, t_local, (int32_t)4, processNum);
            Add(off_local, off_local, t_local, processNum);
            inQueueW.EnQue(w_local);
            inQueueW.DeQue<float>();
            Gather(w_local, w_local, t_local.ReinterpretCast<uint32_t>(), (uint32_t)0, processNum);
            inQueueX.EnQue(x_local);
            inQueueX.DeQue<float>();
            Gather(x_local, x_local, off_local.ReinterpretCast<uint32_t>(), (uint32_t)0, processNum);
            Mul(x_local, x_local, w_local, processNum);
            ReduceSum<float>(y_local, x_local, x_local, processNum);
        }
        y_local.SetValue(0, -1 * y_local.GetValue(0));
        outQueueY.EnQue(y_local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> y_local = outQueueY.DeQue<float>();
        DataCopyExtParams copyParams2 = {1, (uint32_t)(sizeof(float)), 0, 0, 0};
        SetAtomicAdd<float>();
        DataCopyPad(yGm, y_local, copyParams2);
    }
private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueT;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueW;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECCALC, BUFFER_NUM> offQueue;
    GlobalTensor<float> xGm;
    GlobalTensor<int32_t> targetGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> yGm;
    int32_t tensorCount;
    int32_t blockLen;
    int32_t processNum;
    int32_t blockIdx;
};

class KernelNLLLossNone
{
public:
    __aicore__ inline KernelNLLLossNone() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, uint32_t tensorCount,
                                uint32_t blockLen)
    {
        this->tensorCount = tensorCount;   // N
        this->blockLen = blockLen;         // C * d1 * .. * dk
        if (blockLen == 0)
        {
            xGm.SetGlobalBuffer((__gm__ float *)x, tensorCount);
            targetGm.SetGlobalBuffer((__gm__ int32_t *)target, tensorCount);
            weightGm.SetGlobalBuffer((__gm__ float *)weight, tensorCount);
            yGm.SetGlobalBuffer((__gm__ float *)y, 1);
        }
        else
        {
            xGm.SetGlobalBuffer((__gm__ float *)x, tensorCount * blockLen);
            targetGm.SetGlobalBuffer((__gm__ int32_t *)target, tensorCount);
            weightGm.SetGlobalBuffer((__gm__ float *)weight, blockLen);
            yGm.SetGlobalBuffer((__gm__ float *)y, tensorCount);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLen == 0)
        {
            int target = targetGm.GetValue(0);
            float x = xGm.GetValue(target);
            float y = -1 * x;
            yGm.SetValue(0, y);
        }
        else 
        {
            float sum_x = 0, sum_w = 0;
            for (int i = 0; i < tensorCount; i++)
            {
                int k = targetGm.GetValue(i);
                float w = weightGm.GetValue(k);
                float x = xGm.GetValue(i * blockLen + k);
                yGm.SetValue(i, -1 * x * w);
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<int32_t> targetGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> yGm;
    uint32_t tensorCount;
    uint32_t blockLen;
    uint32_t processNum;
};

extern "C" __global__ __aicore__ void nll_loss(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if (TILING_KEY_IS(1)) // mean
    {
        KernelNLLLossMean op;
        op.Init(x, target, weight, y, tiling_data.tensorCount, tiling_data.blockLen);
        op.Process();
    }
    else if (TILING_KEY_IS(2)) // sum
    {
        KernelNLLLossSum op;
        TPipe pipe;
        op.Init(x, target, weight, y, tiling_data.tensorCount, tiling_data.blockLen, tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreProcessNum, &pipe);
        op.Process();
    }
    else if (TILING_KEY_IS(3)) // none
    {
        KernelNLLLossNone op;
        op.Init(x, target, weight, y, tiling_data.tensorCount, tiling_data.blockLen);
        op.Process();
    }
}