# Train/Test 분포 비교 리포트

생성일시: 2026-01-29 02:59:14

## 요약
- **Train**: 24593개 (95.3%)
- **Test**: 1202개 (4.7%)
- **검증 기준**: 분포 차이 5.0% 이내

## 축별 분포 비교

### context.duration
- 최대 차이: 2.73% ✅ PASS

| 값 | Train (%) | Test (%) | 차이 |
|---|---|---|---|
| Long-term course (1-6 months) | 35.21 | 37.94 | 2.73  |
| Mid-term course (2-4 weeks) | 35.52 | 34.53 | 0.99  |
| Short-term intensive (within 1 week) | 29.27 | 27.54 | 1.73  |

### context.learning_environment
- 최대 차이: 1.09% ✅ PASS

| 값 | Train (%) | Test (%) | 차이 |
|---|---|---|---|
| Blended learning | 14.37 | 14.14 | 0.23  |
| Mobile microlearning | 13.74 | 13.56 | 0.18  |
| Offline (classroom) | 12.03 | 11.81 | 0.22  |
| Online asynchronous (LMS) | 20.37 | 21.46 | 1.09  |
| Online synchronous (Zoom, etc.) | 12.75 | 12.81 | 0.06  |
| Project-based learning (PBL) | 13.8 | 13.64 | 0.16  |
| Simulation/VR-based | 12.94 | 12.56 | 0.38  |

### context.class_size
- 최대 차이: 1.49% ✅ PASS

| 값 | Train (%) | Test (%) | 차이 |
|---|---|---|---|
| Large (30+ learners) | 46.6 | 48.09 | 1.49  |
| Medium (10-30 learners) | 25.47 | 24.04 | 1.43  |
| Small (1-10 learners) | 27.93 | 27.87 | 0.06  |

### context.institution_type
- 최대 차이: 2.5% ✅ PASS

| 값 | Train (%) | Test (%) | 차이 |
|---|---|---|---|
| Corporate/Enterprise | 20.54 | 23.04 | 2.5  |
| Graduate school | 12.16 | 10.73 | 1.43  |
| K-12 school | 12.26 | 12.73 | 0.47  |
| Public/Non-profit educational institution | 19.75 | 19.55 | 0.2  |
| University (undergraduate) | 17.28 | 18.14 | 0.86  |
| Vocational training institution | 18.01 | 15.81 | 2.2  |

### context.learner_age
- 최대 차이: 3.54% ✅ PASS

| 값 | Train (%) | Test (%) | 차이 |
|---|---|---|---|
| 40s and above | 23.42 | 26.96 | 3.54  |
| In their 20s | 35.79 | 32.45 | 3.34  |
| In their 30s | 26.36 | 23.79 | 2.57  |
| Teens (13-19) | 14.43 | 16.81 | 2.38  |

### context.learner_education
- 최대 차이: 1.23% ✅ PASS

| 값 | Train (%) | Test (%) | 차이 |
|---|---|---|---|
| Adult learner (non-degree) | 27.9 | 27.62 | 0.28  |
| Elementary | 9.23 | 9.9 | 0.67  |
| High school | 20.74 | 19.55 | 1.19  |
| Middle school | 17.15 | 16.72 | 0.43  |
| University | 24.98 | 26.21 | 1.23  |

### context.domain_expertise
- 최대 차이: 0.7% ✅ PASS

| 값 | Train (%) | Test (%) | 차이 |
|---|---|---|---|
| Advanced | 14.01 | 14.39 | 0.38  |
| Beginner | 43.35 | 43.68 | 0.33  |
| Intermediate | 42.63 | 41.93 | 0.7  |

### domain
- 최대 차이: 0.69% ✅ PASS

| 값 | Train (%) | Test (%) | 차이 |
|---|---|---|---|
| AI | 3.98 | 3.41 | 0.57  |
| Business/HR/Admin Support | 3.97 | 3.66 | 0.31  |
| Education (Teaching & Learning) | 4.29 | 4.08 | 0.21  |
| Language | 19.19 | 19.8 | 0.61  |
| Mathematics | 18.87 | 19.47 | 0.6  |
| Medical/Nursing | 3.98 | 3.49 | 0.49  |
| Science | 18.94 | 19.63 | 0.69  |
| Service/Customer Support | 3.98 | 3.49 | 0.49  |
| Social Studies | 18.81 | 19.47 | 0.66  |
| Software Development/IT | 3.98 | 3.49 | 0.49  |

### difficulty
- 최대 차이: 0.32% ✅ PASS

| 값 | Train (%) | Test (%) | 차이 |
|---|---|---|---|
| Easy - Simple structure with minimal constraints | 32.99 | 33.03 | 0.04  |
| Hard - Complex requirements with multiple constraints | 32.99 | 33.28 | 0.29  |
| Moderate - Standard complexity with some constraints | 34.01 | 33.69 | 0.32  |

## 전체 검증 결과
✅ **모든 축에서 분포 차이가 5% 이내입니다.**