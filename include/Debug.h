#pragma once
#include <iostream>

#ifdef _DEBUG
#define PRINT_SIZE(mat)	std::cout<<(#mat)<<": ("<<mat.rows()<<", "<<mat.cols()<<")\n";
#else
#define PRINT_SIZE(mat)
#endif

namespace engine {

}
