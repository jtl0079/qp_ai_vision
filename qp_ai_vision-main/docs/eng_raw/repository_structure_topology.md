

```
Project/
├───docs                        
│   └───eng_raw                        
├───notebooks                   # .jpynb
├───outputs                     # Generated, downloaded, and temporary artifacts.
│   ├───datasets                # Dataset resources.
│   │   ├───downloads           # Original datasets downloaded from external sources.
│   │   │   ├───kaggle
│   │   │   │   └───sshikamaru_car_object_detection
│   │   │   ├───roboflow
│   │   │   └───manual
│   │   │
│   │   ├───yolo                # Generated YOLO datasets.
│   │   ├───coco                # Generated COCO datasets.
│   │   └───voc                 # Generated Pascal VOC datasets.
│   │
│   ├───experiments/
│   ├───logs                    # Runtime logs.
│   └───temp                    # Temporary files that can be safely removed.
|
├───src
│   ├───rsw_ai
│   │   ├───api
│   │   ├───backend
│   │   ├───enum
│   │   ├───interface
│   │   ├───model
│   │   ├───pipeline
│   │   │   └───__pycache__
│   │   └───__pycache__
│   └───rsw_ai.egg-info
└───tools
```