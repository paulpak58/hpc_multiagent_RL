#include "board_array_access.h"

int access_array_index(const int* index,const int* dim,const int dimensionality){
	int flattened_pos = 0;
	int multiplier = 1;
	for(int i=dimensionality-1;i>=0;i--){
		flattened_pos += index_arr[i]*multiplier;
		multiplier *= dim_arr[i];
	}
	return flattened_pos;
}

