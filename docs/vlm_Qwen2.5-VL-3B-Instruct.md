Model: Qwen/Qwen2.5-VL-3B-Instruct

Reading drawings: 371/427 sections read exactly (86.9%). FerroSight's parser: 100%.

Judging 426 members (all planted defects + a sample of correct members):

| System | Missing caught | Moved caught | Wrong size caught | False alarms on correct members | Left for review |
| --- | ---: | ---: | ---: | ---: | ---: |
| Geometric (FerroSight) | 7/9 | 10/12 | 9/11 | 0/394 (0.0%) | 8 |
| VLM only | 0/9 | 0/12 | 10/11 | 345/394 (87.6%) | 0 |
| Hybrid (VLM decides the review cases) | 7/9 | 10/12 | 10/11 | 6/394 (1.5%) | 0 |
