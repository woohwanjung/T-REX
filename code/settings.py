import os
use_line_profiler = False


DEBUG_NOSHUFFLE = False
HALF_PRECISION = False
DEBUG_NOSAVE = False
DEBUG_DIMENSION_MATCHING =  True
WRITER_DIR = "../tb_log"


WRITER_DIR_DUAL = f"{WRITER_DIR}/dual"
WRITER_DIR_TITLE = f"{WRITER_DIR}/dual_title"
WRITER_DIR_PAIRWISE = f"{WRITER_DIR}/dual_pairwise"

EXT_DIR = "../res"


EXTRACTION_DIR = f"{EXT_DIR}/extracted"
TEST_RESULT_DIR = f"{EXT_DIR}/test_result"
CHECKPOINT_DIR = f"{EXT_DIR}/checkpoint"
CONFIDENCE_DIR = f"{EXT_DIR}/confidence"
ENSEMBLE_DIR = f"{EXT_DIR}/ensemble"

FIGURE_DIR = "../fig_result"

def conditional_profiler(func):
    if use_line_profiler:
        return profile(func)
    return func



if __name__=="__main__":
    directory_list = [EXT_DIR, WRITER_DIR_TITLE, CHECKPOINT_DIR, CONFIDENCE_DIR, WRITER_DIR_PAIRWISE, WRITER_DIR_DUAL]
    for dirpath in directory_list:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)


