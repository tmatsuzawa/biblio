import numpy as np
import process_data as process

arr = np.arange(30).reshape(5,6).astype(float)
arr[0,2], arr[3,3] = np.nan, np.nan

mask = process.get_mask_for_nan_and_inf(arr)
arr_interp = process.interpolate_using_mask(arr, mask, method='linear')
