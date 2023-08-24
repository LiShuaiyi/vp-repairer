# 1-3
rule = "R_G1"
# mode = 'initial'
mode = "repaired"

import ffmpeg

(
    ffmpeg.input(
        "/home/yuanfei/commonroad/commonroad_repairer/tests/case_study/figures/"
        + "/rg"
        + rule[-1]
        + "_"
        + mode
        + "/*.svg",
        pattern_type="glob",
        framerate=10,
    )
    .output(rule + mode + ".mp4")
    .run()
)
