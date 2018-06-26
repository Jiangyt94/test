//
//  main.c
//  test
//
//  Created by Yuntao Jiang on 2018/6/26.
//  Copyright © 2018年 jyt94. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>
#include "mykernel.cl.h"

#define NUM_VALUES 1024

static int validate(cl_float* input, cl_float* output){
    int i;
    for(i=0; i<NUM_VALUES; i++){
        if(output[i] != (input[i]*input[i])){
            fprintf(stdout,
                    "Error element %d did not match", i);
            fflush(stdout);
            return 0;
        }
    }
    return 1;
}

static void print_device_info(cl_device_id device){
    char name[128];
    char vendor[128];
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 128, vendor, NULL);
    fprintf(stdout, "%s : %s\n", vendor, name);
}

int main(int argc, const char * argv[]) {
    int i;
    char name[128];
    dispatch_queue_t queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    if(queue==NULL){
        queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
    cl_device_id gpu = gcl_get_device_id_with_dispatch_queue(queue);
    clGetDeviceInfo(gpu, CL_DEVICE_NAME, 128, name, NULL);
    fprintf(stdout, "Created a dispatch queue using the %s_n\n", name);
    
    float* test_in = (float*) malloc(sizeof(cl_float)*NUM_VALUES);
    for(i=0; i<NUM_VALUES; i++){
        test_in[i] = (cl_float)i;
    }
    float* test_out =  (float*)malloc(sizeof(cl_float)*NUM_VALUES);
    void* mem_in = gcl_malloc(sizeof(cl_float)*NUM_VALUES, test_in, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
    void* mem_out = gcl_malloc(sizeof(cl_float)*NUM_VALUES, NULL, CL_MEM_WRITE_ONLY);
    dispatch_sync(queue, ^{
        size_t wgs;
        gcl_get_kernel_block_workgroup_info(square_kernel, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL);
        cl_ndrange range = {
            1, //the number of dimensions to use
            {0,0,0}, //offset
            {NUM_VALUES, 0,0}, //global range
            {wgs, 0,0}
        };
        
        //ndrange as the first parameter
        square_kernel(&range, (cl_float*)mem_in, (cl_float*)mem_out);
        gcl_memcpy(test_out, mem_out, sizeof(cl_float)*NUM_VALUES);
    });
    if(validate(test_in, test_out)){
        fprintf(stdout, "All values were properly squared.\n");
    }
    
    gcl_free(mem_in);
    gcl_free(mem_out);
    free(test_in);
    free(test_out);
    dispatch_release(queue);
    return 0;
}
