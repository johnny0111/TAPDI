__kernel void multiply(__global int* arr, int c)
{
int i = get_global_id(0);
arr[i] = arr[i] * c;
}