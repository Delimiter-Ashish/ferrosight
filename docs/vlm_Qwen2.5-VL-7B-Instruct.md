Model: Qwen/Qwen2.5-VL-7B-Instruct

Reading drawings: 375/427 sections read exactly (87.8%). FerroSight's parser: 100%.

Judging 426 members (all planted defects + a sample of correct members):

| System | Missing caught | Moved caught | Wrong size caught | False alarms on correct members | Left for review |
| --- | ---: | ---: | ---: | ---: | ---: |
| Geometric (FerroSight) | 7/9 | 10/12 | 9/11 | 0/394 (0.0%) | 8 |
| VLM only | 4/9 | 6/12 | 10/11 | 394/394 (100.0%) | 0 |
| Hybrid (VLM decides the review cases) | 7/9 | 10/12 | 9/11 | 7/394 (1.8%) | 0 |
