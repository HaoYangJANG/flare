"""Fixed authenticated45k continuation; uses the existing guarded native runner."""
from pathlib import Path
from tools import resume_best_bin_long as resume


def configure():
    resume.SOURCE = Path('/root/autodl-tmp/exps/bin_best_multiscale_long_20261004/train/model_0044999.pth')
    resume.SOURCE_SHA = '8b694abd32068075d3231afb10860fd701ace885207ad86d85f42d43561366a2'
    resume.OUTPUT = Path('/root/autodl-tmp/exps/bin_best_multiscale_resume45k_20261006_1122')
    resume.MAX_ITER = 200000
    resume.SOURCE_LOOPS = 45000
    resume.SOURCE_APPLIED = 44969
    resume.SOURCE_SKIPPED = 31
    resume.SOURCE_SCALER = dict(scale=1., growth_factor=2., backoff_factor=.5, growth_interval=2000, _growth_tracker=137)
    resume.LAUNCHER = resume.run.ROOT / 'docs/handoff/run_best_bin_resume45k_20261006_1122.sh'
    resume.EXTRA_IMMUTABLE = (Path('/mnt/c/Users/jianghaoyang/Documents/ChatGPT/DIFFUDETR/DiffuDETR-main3/DiffuDETR-main/docs/handoff/start_best_bin_resume45k_20261006_1122.ps1'),)


def main(argv=None):
    configure()
    resume.main(argv)


if __name__ == '__main__':
    main()
