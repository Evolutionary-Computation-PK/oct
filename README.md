# OCT (Optical Coherence Tomography)  
OCT (Optical Coherence Tomography) is a non-invasive imaging technique that provides high-resolution cross-sectional images of tissue structures using infrared light (800–1300 nm). It operates on the principle of low-coherence interferometry, where light is reflected from different tissue layers, and the time delay of these reflections is used to construct a tissue image.

The generated images, called B-scans, are cross-sectional views, with each vertical line representing a depth measurement (A-scan). Multiple A-scans are combined to create a 2D representation of the tissue.

![desc_oct.png](analysis/desc_oct.png)
<br>Image source: https://eyeguru.org/essentials/interpreting-octs/

## OCTMNIST
The **OCTMNIST** dataset is a collection of retinal images obtained using Optical Coherence Tomography (OCT), designed for the task of classification in the context of eye diseases. It consists of OCT scans, each labeled into one of four classes corresponding to different retinal conditions:

1. **Class 0 (Choroidal Neovascularization - CNV)**: Characterized by abnormal blood vessel growth beneath the retina, commonly associated with conditions like wet age-related macular degeneration (AMD). It can lead to vision distortion and loss if left untreated.
   
2. **Class 1 (Diabetic Macular Edema - DME)**: Refers to the swelling of the macula, a part of the retina, due to diabetes. This condition leads to central vision impairment and is a common cause of blindness in diabetic patients.

3. **Class 2 (Drusen)**: Depicts the presence of small yellow deposits beneath the retina, often related to age-related macular degeneration (AMD). Drusen can affect vision, but in the early stages, the condition may be asymptomatic.

4. **Class 3 (Normal)**: Represents healthy, unaffected retinal scans with no visible signs of disease. This class serves as the baseline for distinguishing between pathological conditions.

| Class | Description                  | Number      | Percentage share |
|-------|------------------------------|-------------|--------------------|
| 0     | Choroidal neovascularization | 37,455      | 34.26%             |
| 1     | Diabetic macular edema       | 11,598      | 10.61%             |
| 2     | Drusen                       | 8,866       | 8.11%              |
| 3     | Normal                       | 51,390      | 47.02%             |
| **–** | **Total**                  | **109,309** | **100.00%**        |

![classes_examples.png](analysis/classes_examples.png)

## Dataset Source
https://medmnist.com/ <br>
code: https://github.com/MedMNIST/MedMNIST <br>
dataset: https://zenodo.org/records/10519652 <br>
paper: https://www.nature.com/articles/s41597-022-01721-8 <br>

Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.

Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.
