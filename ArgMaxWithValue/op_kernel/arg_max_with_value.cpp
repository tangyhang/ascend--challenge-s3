#include "kernel_operator.h"
#include <limits>
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelArgMaxWithValuehalf
{
public:
    __aicore__ inline KernelArgMaxWithValuehalf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, uint32_t reduceAxisLen, uint32_t subTensorLen, uint32_t outputLength,
                                uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreNum, uint32_t smallCoreProcessNum)
    {
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreProcessNum * GetBlockIdx();
        this->reduceAxisLen = reduceAxisLen;
        this->subTensorLen = subTensorLen;
        this->outputLength = outputLength;
        if (coreNum < bigCoreNum)
        {
            this->processNum = bigCoreProcessNum;
        }
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (GetBlockIdx() - bigCoreNum);
        }
        xGm.SetGlobalBuffer((__gm__ half *)x + globalBufferIndex * reduceAxisLen * subTensorLen, reduceAxisLen * subTensorLen * processNum);
        yGm.SetGlobalBuffer((__gm__ int32_t *)indice + globalBufferIndex * subTensorLen, subTensorLen * processNum);
        zGm.SetGlobalBuffer((__gm__ half *)values + globalBufferIndex * subTensorLen, subTensorLen * processNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, (reduceAxisLen * subTensorLen * sizeof(half) + 31) / 32 * 32);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, (subTensorLen * sizeof(int32_t) + 31) / 32 * 32);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, (subTensorLen * sizeof(half) + 31) / 32 * 32);
        pipe.InitBuffer(workQueue1, BUFFER_NUM, (reduceAxisLen * sizeof(uint32_t) + 31) / 32 * 32);
        pipe.InitBuffer(workQueue2, BUFFER_NUM, (reduceAxisLen * sizeof(half) + 31) / 32 * 32);
        pipe.InitBuffer(workQueue3, BUFFER_NUM, (reduceAxisLen * sizeof(half) + 31) / 32 * 32);
        pipe.InitBuffer(workQueue4, BUFFER_NUM, (reduceAxisLen * sizeof(half) + 31) / 32 * 32);
    }
    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->processNum;
        for (uint32_t i = 0; i < loopCount; i++)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> x_local = inQueueX.AllocTensor<half>();
        uint32_t copysz = this->reduceAxisLen * this->subTensorLen * sizeof(half);
        DataCopyExtParams copyParams{1, copysz, 0, 0, 0};
        DataCopyPadExtParams<half> padParams{true, 0, 0, 0};
        DataCopyPad(x_local, xGm[progress * this->reduceAxisLen * this->subTensorLen], copyParams, padParams);
        // DumpTensor(x_local,5, copysz);
        inQueueX.EnQue(x_local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> x_local = inQueueX.DeQue<half>();
        LocalTensor<int32_t> y_local = outQueueY.AllocTensor<int32_t>();
        LocalTensor<half> z_local = outQueueZ.AllocTensor<half>();
        LocalTensor<uint32_t> off_local = workQueue1.AllocTensor<uint32_t>();
        LocalTensor<half> src_local = workQueue2.AllocTensor<half>();
        LocalTensor<half> dst_local = workQueue3.AllocTensor<half>();
        LocalTensor<half> work_local = workQueue4.AllocTensor<half>();
        for (size_t i = 0; i < reduceAxisLen; i++)
        {
            off_local.SetValue(i, i * subTensorLen * sizeof(half));
        }
        for (size_t i = 0; i < subTensorLen; i++)
        {
            Gather(src_local, x_local, off_local, (uint32_t)(i * sizeof(half)), reduceAxisLen);
            PipeBarrier<PIPE_V>();
            ReduceMax<half>(dst_local, src_local, work_local, reduceAxisLen, true);
            half mx = dst_local.GetValue(0);
            half index_half = dst_local.GetValue(1);
            int32_t mx_index = *reinterpret_cast<int32_t *>(&index_half);
            y_local.SetValue(i, (int32_t)mx_index);
            z_local.SetValue(i, mx);
        }
        outQueueY.EnQue(y_local);
        outQueueZ.EnQue(z_local);
        inQueueX.FreeTensor(x_local);
        workQueue1.FreeTensor(off_local);
        workQueue2.FreeTensor(src_local);
        workQueue3.FreeTensor(dst_local);
        workQueue4.FreeTensor(work_local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<int32_t> y_local = outQueueY.DeQue<int32_t>();
        LocalTensor<half> z_local = outQueueZ.DeQue<half>();
        uint32_t copysz1 = this->subTensorLen * sizeof(int32_t);
        uint32_t copysz2 = this->subTensorLen * sizeof(half);
        DataCopyExtParams copyParams1{1, copysz1, 0, 0, 0};
        DataCopyExtParams copyParams2{1, copysz2, 0, 0, 0};
        DataCopyPadExtParams<half> padParams1{true, 0, 0, 0};
        DataCopyPad(yGm[progress * this->subTensorLen], y_local, copyParams1);
        DataCopyPad(zGm[progress * this->subTensorLen], z_local, copyParams2);
        outQueueY.FreeTensor(y_local);
        outQueueZ.FreeTensor(z_local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue1;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue2;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue3;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue4;
    GlobalTensor<half> xGm;
    GlobalTensor<int32_t> yGm;
    GlobalTensor<half> zGm;
    uint32_t reduceAxisLen;
    uint32_t subTensorLen;
    uint32_t outputLength;
    uint32_t processNum;
};

class KernelArgMaxWithValuefloat
{
public:
    __aicore__ inline KernelArgMaxWithValuefloat() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, uint32_t reduceAxisLen, uint32_t subTensorLen, uint32_t outputLength,
                                uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreNum, uint32_t smallCoreProcessNum, uint32_t singleProcessNum, uint32_t bigTileNum, uint32_t bigTailNum, uint32_t smallTileNum, uint32_t smallTailNum)
    {
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreProcessNum * GetBlockIdx();
        this->reduceAxisLen = reduceAxisLen;
        this->subTensorLen = subTensorLen;
        this->outputLength = outputLength;
        this->singleProcessNum = singleProcessNum;
        if (coreNum < bigCoreNum)
        {
            this->processNum = bigCoreProcessNum;
            this->tileNum = bigTileNum;
            this->tailProcessNum = bigTailNum;
        }
        else
        {
            this->processNum = smallCoreProcessNum;
            this->tileNum = smallTileNum;
            this->tailProcessNum = smallTailNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (GetBlockIdx() - bigCoreNum);
        }
        
        xGm.SetGlobalBuffer((__gm__ float *)x + globalBufferIndex * reduceAxisLen * subTensorLen, reduceAxisLen * subTensorLen * processNum);
        yGm.SetGlobalBuffer((__gm__ int32_t *)indice + globalBufferIndex * subTensorLen, subTensorLen * processNum);
        zGm.SetGlobalBuffer((__gm__ float *)values + globalBufferIndex * subTensorLen, subTensorLen * processNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->singleProcessNum * ((reduceAxisLen * subTensorLen * sizeof(float) + 31) / 32 * 32));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->singleProcessNum * ((subTensorLen * sizeof(int32_t) + 31) / 32 * 32));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->singleProcessNum * ((subTensorLen * sizeof(float) + 31) / 32 * 32));
        pipe.InitBuffer(workQueue1, BUFFER_NUM, this->singleProcessNum * ((reduceAxisLen * sizeof(uint32_t) + 31) / 32 * 32));
        pipe.InitBuffer(workQueue2, BUFFER_NUM, this->singleProcessNum * ((reduceAxisLen * sizeof(float) + 31) / 32 * 32));
        pipe.InitBuffer(workQueue3, BUFFER_NUM, this->singleProcessNum * ((reduceAxisLen * sizeof(float) + 31) / 32 * 32));
        pipe.InitBuffer(workQueue4, BUFFER_NUM, this->singleProcessNum * ((reduceAxisLen * sizeof(float) + 31) / 32 * 32));
    }
    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->tileNum;
        this->curProcessNum = this->singleProcessNum;
        for (uint32_t i = 0; i < loopCount; i++)
        {
            if(i == this->tileNum - 1){
                this->curProcessNum = this->tailProcessNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<float> x_local = inQueueX.AllocTensor<float>();
        uint32_t copysz = this->reduceAxisLen * this->subTensorLen * sizeof(float);
        DataCopyExtParams copyParams{static_cast<uint16_t>(this->curProcessNum), copysz, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        DataCopyPad(x_local, xGm[progress * this->reduceAxisLen * this->subTensorLen * this->singleProcessNum], copyParams, padParams);
        // DumpTensor(x_local,5, copysz);
        inQueueX.EnQue(x_local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> x_local = inQueueX.DeQue<float>();
        LocalTensor<int32_t> y_local = outQueueY.AllocTensor<int32_t>();
        LocalTensor<float> z_local = outQueueZ.AllocTensor<float>();
        LocalTensor<uint32_t> off_local = workQueue1.AllocTensor<uint32_t>();
        LocalTensor<float> src_local = workQueue2.AllocTensor<float>();
        LocalTensor<float> dst_local = workQueue3.AllocTensor<float>();
        LocalTensor<float> work_local = workQueue4.AllocTensor<float>();
        for (size_t j = 0; j < this->curProcessNum; j ++)
        {
            for (size_t i = 0; i < reduceAxisLen; i++)
            {
                off_local.SetValue(i, i * subTensorLen * sizeof(float));
            }
            for (size_t i = 0; i < subTensorLen; i++)
            {
                Gather(src_local, x_local[j * ((reduceAxisLen * subTensorLen * sizeof(float) + 31) / 32 * 32 / sizeof(float))], off_local, (uint32_t)(i * sizeof(float)), reduceAxisLen);
                PipeBarrier<PIPE_V>();
                ReduceMax<float>(dst_local, src_local, work_local, reduceAxisLen, true);
                float mx = dst_local.GetValue(0);
                float index_half = dst_local.GetValue(1);
                int32_t mx_index = *reinterpret_cast<int32_t *>(&index_half);
                y_local.SetValue(i + j * subTensorLen, (int32_t)mx_index);
                z_local.SetValue(i + j * subTensorLen, mx);
            }
        }
        outQueueY.EnQue(y_local);
        outQueueZ.EnQue(z_local);
        inQueueX.FreeTensor(x_local);
        workQueue1.FreeTensor(off_local);
        workQueue2.FreeTensor(src_local);
        workQueue3.FreeTensor(dst_local);
        workQueue4.FreeTensor(work_local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<int32_t> y_local = outQueueY.DeQue<int32_t>();
        LocalTensor<float> z_local = outQueueZ.DeQue<float>();
        uint32_t copysz1 = this->subTensorLen * sizeof(int32_t) * this->curProcessNum;
        uint32_t copysz2 = this->subTensorLen * sizeof(float) * this->curProcessNum;
        DataCopyExtParams copyParams1{1, copysz1, 0, 0, 0};
        DataCopyExtParams copyParams2{1, copysz2, 0, 0, 0};
        DataCopyPadExtParams<float> padParams1{true, 0, 0, 0};
        DataCopyPad(yGm[progress * this->subTensorLen * this->singleProcessNum], y_local, copyParams1);
        DataCopyPad(zGm[progress * this->subTensorLen * this->singleProcessNum], z_local, copyParams2);
        outQueueY.FreeTensor(y_local);
        outQueueZ.FreeTensor(z_local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue1;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue2;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue3;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue4;
    GlobalTensor<float> xGm;
    GlobalTensor<int32_t> yGm;
    GlobalTensor<float> zGm;
    uint32_t reduceAxisLen;
    uint32_t subTensorLen;
    uint32_t outputLength;
    uint32_t processNum;
    uint32_t tileNum;
    uint32_t singleProcessNum;
    uint32_t tailProcessNum;
    uint32_t curProcessNum;
};

class KernelArgMaxWithValueInt32
{
public:
    __aicore__ inline KernelArgMaxWithValueInt32() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, uint32_t reduceAxisLen, uint32_t subTensorLen, uint32_t outputLength,
                                uint32_t bigCoreNum, uint32_t bigCoreProcessNum, uint32_t smallCoreNum, uint32_t smallCoreProcessNum)
    {
        uint32_t coreNum = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreProcessNum * GetBlockIdx();
        this->reduceAxisLen = reduceAxisLen;
        this->subTensorLen = subTensorLen;
        this->outputLength = outputLength;
        if (coreNum < bigCoreNum)
        {
            this->processNum = bigCoreProcessNum;
        }
        else
        {
            this->processNum = smallCoreProcessNum;
            globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (GetBlockIdx() - bigCoreNum);
        }
        xGm.SetGlobalBuffer((__gm__ int32_t *)x + globalBufferIndex * reduceAxisLen * subTensorLen, reduceAxisLen * subTensorLen * processNum);
        yGm.SetGlobalBuffer((__gm__ int32_t *)indice + globalBufferIndex * subTensorLen, subTensorLen * processNum);
        zGm.SetGlobalBuffer((__gm__ int32_t *)values + globalBufferIndex * subTensorLen, subTensorLen * processNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, (reduceAxisLen * subTensorLen * sizeof(int32_t) + 31) / 32 * 32);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, (subTensorLen * sizeof(int32_t) + 31) / 32 * 32);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, (subTensorLen * sizeof(int32_t) + 31) / 32 * 32);
        pipe.InitBuffer(workQueue1, BUFFER_NUM, (reduceAxisLen * sizeof(uint32_t) + 31) / 32 * 32);
        pipe.InitBuffer(workQueue2, BUFFER_NUM, (reduceAxisLen * sizeof(int32_t) + 31) / 32 * 32);
        pipe.InitBuffer(workQueue3, BUFFER_NUM, (reduceAxisLen * sizeof(float) + 31) / 32 * 32);
        pipe.InitBuffer(workQueue4, BUFFER_NUM, (reduceAxisLen * sizeof(float) + 31) / 32 * 32);
        pipe.InitBuffer(castQueue, BUFFER_NUM, (reduceAxisLen * sizeof(float) + 31) / 32 * 32);
    }
    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->processNum;
        for (uint32_t i = 0; i < loopCount; i++)
        {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<int32_t> x_local = inQueueX.AllocTensor<int32_t>();
        uint32_t copysz = this->reduceAxisLen * this->subTensorLen * sizeof(int32_t);
        DataCopyExtParams copyParams{1, copysz, 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        DataCopyPad(x_local, xGm[progress * this->reduceAxisLen * this->subTensorLen], copyParams, padParams);
        // DumpTensor(x_local,5, copysz);
        inQueueX.EnQue(x_local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<int32_t> x_local = inQueueX.DeQue<int32_t>();
        LocalTensor<int32_t> y_local = outQueueY.AllocTensor<int32_t>();
        LocalTensor<int32_t> z_local = outQueueZ.AllocTensor<int32_t>();
        LocalTensor<uint32_t> off_local = workQueue1.AllocTensor<uint32_t>();
        LocalTensor<int32_t> src_local = workQueue2.AllocTensor<int32_t>();
        LocalTensor<float> dst_local = workQueue3.AllocTensor<float>();
        LocalTensor<float> work_local = workQueue4.AllocTensor<float>();
        LocalTensor<float> cast_local = castQueue.AllocTensor<float>();
        for (size_t i = 0; i < reduceAxisLen; i++)
        {
            off_local.SetValue(i, i * subTensorLen * sizeof(int32_t));
        }
        for (size_t i = 0; i < subTensorLen; i++)
        {
            Gather(src_local, x_local, off_local, (uint32_t)(i * sizeof(int32_t)), reduceAxisLen);
            PipeBarrier<PIPE_V>();
            Cast(cast_local, src_local, RoundMode::CAST_NONE, reduceAxisLen);
            PipeBarrier<PIPE_V>();
            ReduceMax<float>(dst_local, cast_local, work_local, reduceAxisLen, true);
            float mx = dst_local.GetValue(0);
            float index_half = dst_local.GetValue(1);
            int32_t mx_index = *reinterpret_cast<int32_t *>(&index_half);
            y_local.SetValue(i, (int32_t)mx_index);
            z_local.SetValue(i, (int32_t)mx);
        }
        outQueueY.EnQue(y_local);
        outQueueZ.EnQue(z_local);
        inQueueX.FreeTensor(x_local);
        workQueue1.FreeTensor(off_local);
        workQueue2.FreeTensor(src_local);
        workQueue3.FreeTensor(dst_local);
        workQueue4.FreeTensor(work_local);
        castQueue.FreeTensor(cast_local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<int32_t> y_local = outQueueY.DeQue<int32_t>();
        LocalTensor<int32_t> z_local = outQueueZ.DeQue<int32_t>();
        uint32_t copysz1 = this->subTensorLen * sizeof(int32_t);
        uint32_t copysz2 = this->subTensorLen * sizeof(int32_t);
        DataCopyExtParams copyParams1{1, copysz1, 0, 0, 0};
        DataCopyExtParams copyParams2{1, copysz2, 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams1{true, 0, 0, 0};
        DataCopyPad(yGm[progress * this->subTensorLen], y_local, copyParams1);
        DataCopyPad(zGm[progress * this->subTensorLen], z_local, copyParams2);
        outQueueY.FreeTensor(y_local);
        outQueueZ.FreeTensor(z_local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue1;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue2;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue3;
    TQue<QuePosition::VECCALC, BUFFER_NUM> workQueue4;
    TQue<QuePosition::VECCALC, BUFFER_NUM> castQueue;
    GlobalTensor<int32_t> xGm;
    GlobalTensor<int32_t> yGm;
    GlobalTensor<int32_t> zGm;
    uint32_t reduceAxisLen;
    uint32_t subTensorLen;
    uint32_t outputLength;
    uint32_t processNum;
};


class KernelArgMaxWithValueUint8 {
public:
    __aicore__ inline KernelArgMaxWithValueUint8() {}

   __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, uint32_t reduceAxisLen, uint32_t subTensorLen, uint32_t outputLength)
    {
        this->reduceAxisLen = reduceAxisLen;
        this->subTensorLen = subTensorLen;
        this->outputLength = outputLength;
        xGm.SetGlobalBuffer((__gm__ uint8_t *)x, reduceAxisLen * outputLength + subTensorLen);
        yGm.SetGlobalBuffer((__gm__ int32_t *)indice, outputLength);
        zGm.SetGlobalBuffer((__gm__ uint8_t *)values, outputLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, (reduceAxisLen * subTensorLen * sizeof(uint8_t) + 31) / 32 * 32);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, (subTensorLen * sizeof(int32_t) + 31) / 32 * 32);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, (subTensorLen * sizeof(uint8_t) + 31) / 32 * 32);
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->outputLength / this->subTensorLen;
        for (uint32_t i = 0; i < loopCount; i++)
        {
            for(uint32_t j = 0; j < subTensorLen; j ++)
            {
                uint8_t mx = xGm.GetValue(i * reduceAxisLen * subTensorLen + j);
                int32_t mx_index = 0;
                for(uint32_t k = 1; k < reduceAxisLen; k ++)
                {
                    uint8_t val = xGm.GetValue(i * reduceAxisLen * subTensorLen + k * subTensorLen + j);
                    int32_t mx_ = (int32_t)mx;
                    int32_t val_ = (int32_t)val;
                    if(val_ > mx_)
                    {
                        mx = val;
                        mx_index = k;
                    }
                }
                yGm.SetValue(i * reduceAxisLen * subTensorLen + j, mx_index);
                zGm.SetValue(i * reduceAxisLen * subTensorLen + j, mx);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<uint8_t> x_local = inQueueX.AllocTensor<uint8_t>();
        uint32_t copysz = this->reduceAxisLen * this->subTensorLen * sizeof(uint8_t);
        DataCopyExtParams copyParams{1, copysz, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{true, 0, 0, 0};
        DataCopyPad(x_local, xGm[progress * this->reduceAxisLen * this->subTensorLen], copyParams, padParams);
        // DumpTensor(x_local,5, copysz);
        inQueueX.EnQue(x_local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<uint8_t> x_local = inQueueX.DeQue<uint8_t>();
        LocalTensor<int32_t> y_local = outQueueY.AllocTensor<int32_t>();
        LocalTensor<uint8_t> z_local = outQueueZ.AllocTensor<uint8_t>();
        for(size_t i = 0; i < subTensorLen; i ++)
        {
            uint8_t mx = x_local.GetValue(i);
            int32_t mx_index = 0;
            for(size_t j = 1; j < reduceAxisLen; j ++)
            {
                uint8_t val = x_local.GetValue(j * subTensorLen + i);
                int32_t mx_ = (int32_t)mx;
                int32_t val_ = (int32_t)val;
                if(val_ > mx_)
                {
                    mx = val;
                    mx_index = j;
                }
            }
            y_local.SetValue(i, mx_index);
            z_local.SetValue(i, mx);
        }
        //DumpTensor(y_local,5, 32);
        outQueueY.EnQue(y_local);
        outQueueZ.EnQue(z_local);
        inQueueX.FreeTensor(x_local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<int32_t> y_local = outQueueY.DeQue<int32_t>();
        LocalTensor<uint8_t> z_local = outQueueZ.DeQue<uint8_t>();
        uint32_t copysz1 = this->subTensorLen * sizeof(int32_t);
        uint32_t copysz2 = this->subTensorLen * sizeof(uint8_t);
        DataCopyExtParams copyParams1{1, copysz1, 0, 0, 0};
        DataCopyExtParams copyParams2{1, copysz2, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams1{true, 0, 0, 0};
        DataCopyPad(yGm[progress * this->subTensorLen], y_local, copyParams1);
        DataCopyPad(zGm[progress * this->subTensorLen], z_local, copyParams2);
        outQueueY.FreeTensor(y_local);
        outQueueZ.FreeTensor(z_local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<uint8_t> xGm;
    GlobalTensor<int32_t> yGm;
    GlobalTensor<uint8_t> zGm;
    uint32_t reduceAxisLen;
    uint32_t subTensorLen;
    uint32_t outputLength;
};

extern "C" __global__ __aicore__ void arg_max_with_value(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    if(TILING_KEY_IS(1)) {
        KernelArgMaxWithValueUint8 op;
        op.Init(x, indice, values, tiling_data.reduceAxisLen, tiling_data.subTensorLen, tiling_data.outputLength);
        op.Process();
    }
    else if(TILING_KEY_IS(2)) {
        KernelArgMaxWithValueInt32 op;
        op.Init(x, indice, values, tiling_data.reduceAxisLen, tiling_data.subTensorLen, tiling_data.outputLength, 
            tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreNum, tiling_data.smallCoreProcessNum);
        op.Process();
    }
    else if(TILING_KEY_IS(3)) {
        KernelArgMaxWithValuehalf op;
        op.Init(x, indice, values, tiling_data.reduceAxisLen, tiling_data.subTensorLen, tiling_data.outputLength, 
            tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreNum, tiling_data.smallCoreProcessNum);
        op.Process();
    }
    else if(TILING_KEY_IS(4)) {
        KernelArgMaxWithValuefloat op;
        op.Init(x, indice, values, tiling_data.reduceAxisLen, tiling_data.subTensorLen, tiling_data.outputLength, 
            tiling_data.bigCoreNum, tiling_data.bigCoreProcessNum, tiling_data.smallCoreNum, tiling_data.smallCoreProcessNum, tiling_data.singleProcessNum, tiling_data.bigTileNum, tiling_data.bigTailNum, tiling_data.smallTileNum, tiling_data.smallTailNum);
        op.Process();
    }
}
