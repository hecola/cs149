#include <stdio.h>
#include <thread>

#include "CycleTimer.h"

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

static inline int mandel_thread(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {
        float z_re2 = z_re * z_re;
        float z_im2 = z_im * z_im;

        if (z_re2 + z_im2 > 4.f)
            break;

        float new_re = z_re2 - z_im2;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{

    // TODO FOR CS149 STUDENTS: Implement the body of the worker
    // thread here. Each thread should make a call to mandelbrotSerial()
    // to compute a part of the output image.  For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    // double startTime = CycleTimer::currentSeconds();

    // printf("Hello world from thread %d\n", args->threadId);
    const float dx = (args->x1 - args->x0) / args->width;
    const float dy = (args->y1 - args->y0) / args->height;
    const int height = args->height;
    const int width = args->width;

    const int numThreads = args->numThreads;
    const int maxIterations = args->maxIterations;
    const int x0 = args->x0;
    const int y0 = args->y0;
    const int threadId = args->threadId;
    int *output = args->output;
    // for (int j = threadId; j < height; j+=numThreads)
    // {
    //     for (int i = 0; i < width; ++i)
    //     {
    //         float x = x0 + i * dx;
    //         float y = y0 + j * dy;

    //         int index = (j * width + i);
    //         args->output[index] = mandel(x, y, maxIterations);
    //     }
    // }
    // const int space =  1;
    // int thread_step = (numThreads - 1) * space;
    // register float x, y;
    // register int index;
    // for (int j = 0; j < height; j++)
    // {
    //     y = y0 + j * dy;
    //     for (register int count = 0, i = threadId * space; i < width;)
    //     {
    //         // x = x0 + i * dx;
    //         //index = ();
    //         output[j * width + i] = mandel_thread(x0 + i * dx, y, maxIterations);
    //         count++;
    //         i++;
    //         if (count == space) // each thread computes space pixels, then jump to the next space pixels
    //         {
    //             count = 0;
    //             i += thread_step;
    //         }
    //     }
    // }
    for (int j = 0; j < height; j++)
    {
        for (int i = threadId; i < width; i += numThreads)
        {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel_thread(x, y, maxIterations);
        }
    }
    // for (int i = threadId; i < width; i += numThreads)
    // {
    //     for (int j = 0; j < height; j++)
    //     {
    //         float x = x0 + i * dx;
    //         float y = y0 + j * dy;

    //         int index = (j * width + i);
    //         output[index] = mandel(x, y, maxIterations);
    //     }
    // }
    // double endTime = CycleTimer::currentSeconds();
    // printf("threadId: %d, start time: %f, end time: %f, elapsed time: %f ms\n", threadId, startTime, endTime, (endTime - startTime)*1000);
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i = 0; i < numThreads; i++)
    {

        // TODO FOR CS149 STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
