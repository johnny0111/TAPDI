//Sampler
__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
    CLK_ADDRESS_CLAMP_TO_EDGE | //Clamp to zeros
    CLK_FILTER_NEAREST;

//Sobel for BGRA (uchar4) images

__kernel void sobel_BGRA(
    __read_only image2d_t image,
    __write_only image2d_t imageOut,
    int w,
    int h,
    int b,
    float c
    )
{

//Get coordinates
int iX = get_global_id(0);      //column
int iY = get_global_id(1);      //row

if(iX<w && iY<h){
    int x = iX;
    int y = iY;
    uint4 Pixel = read_imageui(image, sampler, (int2)(x , y ));

    uint4 G = (0,0,0,Pixel.w);
    G.x = Pixel.x * c +b;
    G.y = Pixel.y * c +b;
    G.z = Pixel.z * c +b;

    if(G.x > 255)
        G.x = 255;
    if(G.y > 255)
        G.y = 255;
    if(G.z > 255)
        G.z = 255;


    write_imageui( imageOut, (int2)(iX,iY) , G);

}

}