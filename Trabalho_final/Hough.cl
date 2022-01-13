 __constant double cosineTableGlobal[360] = { 1, 0.540302, -0.416147, -0.989992, -0.653644, 0.283662, 0.96017, 0.753902, -0.1455, -0.91113, -0.839072, 0.0044257, 0.843854, 0.907447, 0.136737, -0.759688, -0.957659, -0.275163, 0.660317, 0.988705, 0.408082, -0.547729, -0.999961, -0.532833, 0.424179, 0.991203, 0.646919, -0.292139, -0.962606, -0.748058, 0.154251, 0.914742, 0.834223, -0.0132767, -0.84857, -0.903692, -0.127964, 0.765414, 0.955074, 0.266643, -0.666938, -0.987339, -0.399985, 0.555113, 0.999843, 0.525322, -0.432178, -0.992335, -0.640144, 0.300593, 0.964966, 0.742154, -0.162991, -0.918283, -0.82931, 0.0221268, 0.85322, 0.899867, 0.11918, -0.77108, -0.952413, -0.258102, 0.673507, 0.985897, 0.391857, -0.562454, -0.999647, -0.51777, 0.440143, 0.99339, 0.633319, -0.309023, -0.967251, -0.736193, 0.171717, 0.921751, 0.824331, -0.030975, -0.857803, -0.895971, -0.110387, 0.776686, 0.949678, 0.24954, -0.680023, -0.984377, -0.383698, 0.56975, 0.999373, 0.510177, -0.448074, -0.994367, -0.626444, 0.317429,
0.969459, 0.730174, -0.18043, -0.925148, -0.819288, 0.0398209, 0.862319, 0.892005, 0.101586, -0.782231, -0.946868, -0.240959, 0.686487, 0.98278, 0.37551, -0.577002, -0.999021, -0.502544, 0.455969, 0.995267, 0.619521, -0.32581, -0.971592, -0.724097, 0.189129, 0.928471, 0.814181, -0.0486636, -0.866767, -0.887969, -0.0927762, 0.787715, 0.943984, 0.232359, -0.692896, -0.981106, -0.367291, 0.584209, 0.99859, 0.494872, -0.463829, -0.996088, -0.612548, 0.334165, 0.973649, 0.717964, -0.197814, -0.931722, -0.80901, 0.0575025, 0.871147, 0.883863, 0.0839594, -0.793136, -0.941026, -0.223741, 0.699251, 0.979355, 0.359044, -0.59137, -0.998081, -0.487161, 0.471652, 0.996831, 0.605528, -0.342495, -0.975629, -0.711775, 0.206482, 0.9349, 0.803775, -0.0663369, -0.875459, -0.879689, -0.0751361, 0.798496, 0.937995, 0.215105, -0.705551, -0.977527, -0.350769, 0.598484, 0.997494, 0.479412, -0.479439, -0.997496, -0.59846, 0.350797, 0.977533, 0.70553, -0.215135, -0.938005, -0.798478,
0.0751662, 0.879703, 0.875445, 0.0663069, -0.803793, -0.93489, -0.206453, 0.711796, 0.975623, 0.342466, -0.605552, -0.996829, -0.471626, 0.487188, 0.998083, 0.591345, -0.359072, -0.979361, -0.699229, 0.22377, 0.941037, 0.793118, -0.0839895, -0.883877, -0.871133, -0.0574724, 0.809028, 0.931711, 0.197784, -0.717985, -0.973642, -0.334137, 0.612572, 0.996085, 0.463802, -0.494898, -0.998592, -0.584184, 0.367319, 0.981111, 0.692874, -0.232388, -0.943994, -0.787696, 0.0928062, 0.887983, 0.866752, 0.0486335, -0.814198, -0.92846, -0.1891, 0.724118, 0.971585, 0.325781, -0.619544, -0.995264, -0.455942, 0.50257, 0.999022, 0.576978, -0.375538, -0.982785, -0.686465, 0.240988, 0.946878, 0.782212, -0.101616, -0.892018, -0.862304, -0.0397908, 0.819306, 0.925136, 0.180401, -0.730194, -0.969452, -0.3174, 0.626468, 0.994364, 0.448047, -0.510203, -0.999374, -0.569726, 0.383726, 0.984382, 0.680001, -0.249569, -0.949687, -0.776667, 0.110417, 0.895984, 0.857788, 0.0309449, -0.824348,
-0.92174, -0.171688, 0.736213, 0.967243, 0.308994, -0.633343, -0.993387, -0.440116, 0.517796, 0.999648, 0.562429, -0.391885, -0.985902, -0.673485, 0.258131, 0.952422, 0.771061, -0.11921, -0.89988, -0.853204, -0.0220966, 0.829327, 0.918271, 0.162961, -0.742174, -0.964958, -0.300564, 0.640167, 0.992332, 0.432151, -0.525348, -0.999844, -0.555088, 0.400013, 0.987344, 0.666916, -0.266672, -0.955083, -0.765395, 0.127994, 0.903705, 0.848554, 0.0132466, -0.83424, -0.91473, -0.154222, 0.748078, 0.962598, 0.29211, -0.646942, -0.991199, -0.424152, 0.532859, 0.999961, 0.547704, -0.40811, -0.988709, -0.660294, 0.275192, 0.957668, 0.759668, -0.136767, -0.907459, -0.843838, -0.00439555, 0.839088, 0.911118, 0.14547, -0.753922, -0.960162, -0.283633, 0.653666, 0.989988, 0.416119, -0.540328, -1, -0.540277, 0.416174, 0.989997, 0.653621 };


__constant double sineTableGlobal[360] = { 0, 0.841471, 0.909297, 0.14112, -0.756802, -0.958924, -0.279415, 0.656987, 0.989358, 0.412118, -0.544021, -0.99999, -0.536573, 0.420167, 0.990607, 0.650288, -0.287903, -0.961397, -0.750987, 0.149877, 0.912945, 0.836656, -0.00885131, -0.84622, -0.905578, -0.132352, 0.762558, 0.956376, 0.270906, -0.663634, -0.988032, -0.404038, 0.551427, 0.999912, 0.529083, -0.428183, -0.991779, -0.643538, 0.296369, 0.963795, 0.745113, -0.158623, -0.916522, -0.831775, 0.0177019, 0.850904, 0.901788, 0.123573, -0.768255, -0.953753, -0.262375, 0.670229, 0.986628, 0.395925, -0.558789, -0.999755, -0.521551, 0.436165, 0.992873, 0.636738, -0.304811, -0.966118, -0.739181, 0.167356, 0.920026, 0.826829, -0.0265512, -0.85552, -0.897928, -0.114785, 0.773891, 0.951055, 0.253823, -0.676772, -0.985146, -0.387782, 0.566108, 0.99952, 0.513978, -0.444113, -0.993889, -0.629888, 0.313229, 0.968364, 0.73319, -0.176076, -0.923458, -0.821818, 0.0353983, 0.860069, 0.893997, 0.105988, -0.779466, -0.948282,
-0.245252, 0.683262, 0.983588, 0.379608, -0.573382, -0.999207, -0.506366, 0.452026, 0.994827, 0.622989, -0.321622, -0.970535, -0.727143, 0.184782, 0.926819, 0.816743, -0.0442427, -0.864551, -0.889996, -0.0971819, 0.78498, 0.945435, 0.236661, -0.689698, -0.981952, -0.371404, 0.580611, 0.998815, 0.498713, -0.459903, -0.995687, -0.61604, 0.329991, 0.97263, 0.721038, -0.193473, -0.930106, -0.811603, 0.0530836, 0.868966, 0.885925, 0.0883687, -0.790433, -0.942514, -0.228052, 0.69608, 0.98024, 0.363171, -0.587795, -0.998345, -0.491022, 0.467745, 0.996469, 0.609044, -0.338333, -0.974649, -0.714876, 0.20215, 0.933321, 0.806401, -0.0619203, -0.873312, -0.881785, -0.0795485, 0.795824, 0.93952, 0.219425, -0.702408, -0.97845, -0.35491, 0.594933, 0.997797, 0.483292, -0.47555, -0.997173, -0.602, 0.346649, 0.976591, 0.708659, -0.210811, -0.936462, -0.801135, 0.0707522, 0.87759, 0.877575, 0.0707222, -0.801153, -0.936451, -0.210781, 0.70868, 0.976584, 0.346621, -0.602024, -0.997171,
-0.475524, 0.483318, 0.997799, 0.594909, -0.354938, -0.978457, -0.702386, 0.219455, 0.93953, 0.795806, -0.0795786, -0.881799, -0.873297, -0.0618903, 0.806418, 0.93331, 0.20212, -0.714898, -0.974642, -0.338305, 0.609068, 0.996467, 0.467719, -0.491048, -0.998347, -0.587771, 0.363199, 0.980246, 0.696058, -0.228082, -0.942525, -0.790415, 0.0883987, 0.885939, 0.868951, 0.0530535, -0.811621, -0.930095, -0.193444, 0.721059, 0.972623, 0.329962, -0.616064, -0.995684, -0.459877, 0.498739, 0.998817, 0.580587, -0.371432, -0.981958, -0.689676, 0.236691, 0.945445, 0.784962, -0.0972119, -0.890009, -0.864536, -0.0442126, 0.81676, 0.926807, 0.184752, -0.727163, -0.970528, -0.321594, 0.623012, 0.994824, 0.451999, -0.506392, -0.999208, -0.573357, 0.379636, 0.983593, 0.68324, -0.245281, -0.948292, -0.779447, 0.106017, 0.89401, 0.860054, 0.0353682, -0.821835, -0.923447, -0.176046, 0.733211, 0.968357, 0.3132, -0.629911, -0.993885, -0.444086, 0.514004, 0.999521, 0.566083, -0.387809, -0.985151,
-0.67675, 0.253853, 0.951064, 0.773872, -0.114815, -0.897941, -0.855504, -0.026521, 0.826846, 0.920014, 0.167326, -0.739201, -0.96611, -0.304782, 0.636761, 0.992869, 0.436138, -0.521577, -0.999756, -0.558764, 0.395953, 0.986633, 0.670207, -0.262404, -0.953762, -0.768235, 0.123603, 0.901801, 0.850888, 0.0176718, -0.831791, -0.916509, -0.158593, 0.745133, 0.963787, 0.29634, -0.643561, -0.991775, -0.428155, 0.529108, 0.999912, 0.551402, -0.404065, -0.988036, -0.663611, 0.270935, 0.956385, 0.762539, -0.132382, -0.905591, -0.846204, -0.00882117, 0.836672, 0.912933, 0.149847, -0.751007, -0.961389, -0.287874, 0.650311, 0.990603, 0.42014, -0.536598, -0.99999, -0.543996, 0.412146, 0.989363, 0.656964, -0.279444, -0.958933, -0.756783, 0.14115, 0.90931, 0.841455, -3.01444e-05, -0.841487, -0.909285, -0.14109, 0.756822 };

 kernel void hough_circle(read_only image2d_t imageIn, __global int* accumulator, int w,__global int * circle, int r_min, int r_max, __global int* maxval, int h)
 {
     sampler_t sampler=CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
     int gid0 = get_global_id(0);
     int gid1 = get_global_id(1);
     uint4 pixel;
     int x0=0,y0=0,r;
     pixel=read_imageui(imageIn,sampler,(int2)(gid0,gid1));

    //accumulator[gid0+gid1]=pixel.x;
     if(pixel.x==255)
     {

         for(int r=r_min;r<r_max;r+=2)
         {
            for(int theta=0; theta<360;theta+=2)
            {
                x0=(int) round(gid0-r*cosineTableGlobal[theta]);
                y0=(int) round(gid1-r*sineTableGlobal[theta]);
                if((x0>0) && (x0<get_global_size(0)) && (y0>0)&&(y0<get_global_size(1)))
                    //accumulator[w*y0+x0]+=1;
                    accumulator[x0+y0*w+(r-r_min)*w*h]+=1;
                //if(*maxval<accumulator[w*y0+x0])
                if(*maxval<accumulator[x0+y0*w+(r-r_min)*w*h])
                {
                    //*maxval=accumulator[w*y0+x0];
                    *maxval=accumulator[x0+y0*w+(r-r_min)*w*h];
                    circle[0]=x0;
                    circle[1]=y0;
                    circle[2]=r;
                }
            }

         }
     }

 }