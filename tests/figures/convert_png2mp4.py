# 1-2
rule = "R_G2"
mode = 'initial'
# mode = 'repaired'

import ffmpeg
(
    ffmpeg
    .input('/home/yuanfei/commonroad/commonroad_repair/tests/figures/' +
           '/rg' + rule[-1] + '_' + mode +
           '/*.svg', pattern_type='glob', framerate=10)
    .output(rule + mode + '.mp4')
    .run()
)
