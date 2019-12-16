import os
import glob

def rename_files_in_dir(dir, target, new_str, ext=''):
    """
    Rename files in a directory
    e.g. rename_files_in_dir('.', 'target', 'replaced', ext='txt')
    ./abc_target_efg.txt -> ./abc_replaced_efg.txt

    Parameters
    ----------
    dir
    target
    new_str
    ext

    Returns
    -------

    """
    print('Working directory: %s' % dir)
    if ext == '':
        # List directories and files in dir
        files = os.listdir(dir)
    else:
        files = glob.glob(dir + '/*' + ext)
    for file in files:
        olf_filepath = os.path.join(dir, file)
        if target in file:
            new_file = file.replace(target, new_str)
            new_filepath = os.path.join(dir, new_file)
            os.rename(olf_filepath, new_filepath)
    print('... Replaced %s with %s' % (target, new_str))