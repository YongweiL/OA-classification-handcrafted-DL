# OA-classification-handcrafted-DL
  CLASSIFICATION OF KNEE OSTEOTHRITIS USING  DEEP LEARNING MODELS FUSED WITH  HANDCRAFTED FEATURES  
Citation : 
Mohammed, Nelson, Pooja, Boehm, Ahmed, Mahendrakar, Mahmoudian, Lee, Peat, Cui, Scott, Kellgren, Dube, Kumar, Norman, Thomas, Tan, Taye, Sonavane, Huang, Abd Ghani, Qi, Prashantha, Senan, Olayah, Sedik, Meena, Yunus, Chen, Tiulpin, Wani, Zhang, Meng, Xuan, Kokkotis, Su, Heisinger, Yang, Tiwari, Moustakidis, Ahmed, Wahyuningrum, Nurmirinta
  
In this study, knee OA X-ray pictures in this investigation are classified using Kellgren and Lawrence (KL) grades as the ground truth. When determining the initial severity of knee osteoarthritis on radiographs, the KL grading scheme is still thought to be the best. To represent the radiological severity of knee OA, five grades are used. As shown in Figure below, "Grade 0" denotes normal, "Grade 1" uncertain, "Grade 2" minor, "Grade 3" moderate, and "Grade 4" severe as seen in figure below.
<img width="765" height="345" alt="image" src="https://github.com/user-attachments/assets/89c0829c-8b1c-4385-91dc-c53cfc5a3e47" />

Degenerative knee conditions like osteoarthritis result from the loss of cartilage that cushions joint impact. This study used the OAI dataset to assess KL grading severity and identify knee arthritis from X-rays. The dataset from (https://www.kaggle.com/datasets/tommyngx/kneeoa) includes 9,786 X-rays from 4,796 individuals aged 45–79, covering 8,260 knee joints. Based on the KL grading scheme, images were categorized into five classes: Grade 0 (3,857), Grade 1 (1,770), Grade 2 (2,578), Grade 3 (1,286), and Grade 4 (295). To ensure only clear images were used, a custom Streamlit app was developed for manual selection, resulting in 5,277 high-quality images saved for analysis. Table below summarizes the dataset before and after selection.
<img width="747" height="617" alt="image" src="https://github.com/user-attachments/assets/c321e929-8ac9-49cc-9708-8e99f51d11f9" />

# Proposed Architecture
An end-to-end deep learning pipeline is suggested to diagnose the severity of Knee Osteoarthritis (KOA) and solve a combination of convolutional neural network (CNN) and handcrafted features. During preprocessing, the OAI dataset is preconditioned by means of removing low-quality or redundant pictures based on an auto-selection scheme, and then images are processed by consecutive stages of enhancement, which are CLAHE, resizing, and conversion to grayscale, specific to CNN-based and handcrafted feature extraction tracks. The dataset is in turn grouped into training, validation, and testing group with balanced class distribution per each of the KOA grades. In order to allay the inequality in classes, there will be focused augmentation of the training data only on grade G0 to G4, producing a binary mounded distribution. From two pretrained CNN ( VGG-19 and ResNet-101 ) features are extraced by fetching high-dimensional vectors on their final layers ( 4096 -D in VGG- 19  fc2 and 2048 -D in ResNet- 101 global average pooling layer). At the same time, three generally popular technique are applied to extract handcrafted features based on texture: GLCM (13 features), DWT (12 features), and LBP (203 features), which results in a 228-feature-dimensional vector. Fused representations (3representations) of the extracted CNN features and handcrafted features are obtained at the feature level including; (i) VGG19 + Handcrafted (4324-D), (ii) ResNet101 + Handcrafted (2276-D), and (iii) VGG19 + ResNet101 + Handcrafted (6372-D). These amalgamated vectors are then directed to train three singular Feedforward Neural Network (FFNN) classifiers. The hyperparameter tuning of FFNNs is done through the use of validation data and their final performance in the classification development is measured with the standard metrics including the accuracy, precision, recall, specificity, sensitivity, confusion matrix, and AUC. The comparative analysis shows the most viable combination of fusion strategies of KOA severity classification using automatization.

# Features Extracted Store in CSV
<img width="856" height="419" alt="image" src="https://github.com/user-attachments/assets/8001ac01-6ee6-4097-9919-a5094bbe8612" />

# Feature-Level Fusion - Example Feature Index Map for VGG19 + Handcrafted Fusion (Single Sample)
<img width="639" height="459" alt="image" src="https://github.com/user-attachments/assets/52d255fe-3a87-4309-a802-f68e461c28df" />

# Feature-Level Fusion - Example Feature Index Map for Resnet101 + Handcrafted Fusion (Single Sample)
<img width="856" height="215" alt="image" src="https://github.com/user-attachments/assets/9f01bbd7-6c5b-4120-a293-97f8e11b7ab6" />

# Feature-Level Fusion - Example Feature Index Map for VGG19 + Resnet101 + Handcrafted Fusion (Single Sample)
<img width="856" height="318" alt="image" src="https://github.com/user-attachments/assets/3f3100af-6a52-4dcc-8c18-730380eba2f9" />

# Learning Curve on FFNN Classifier between 3 Models
<img width="703" height="260" alt="image" src="https://github.com/user-attachments/assets/3bc28060-e113-4d4a-b51e-ada901f488f0" />
<img width="723" height="272" alt="image" src="https://github.com/user-attachments/assets/857c2354-38ef-4996-964c-82ff5cb90b37" />
<img width="738" height="238" alt="image" src="https://github.com/user-attachments/assets/d9b12ccf-abaf-49e7-acb1-a8fc44ab53cb" />

# Results Confusion Metric and Comparison between 3 Models
<img width="1255" height="443" alt="image" src="https://github.com/user-attachments/assets/e43cb37e-e3dd-4a49-b64f-3ce6a5f02fcf" />
<img width="596" height="347" alt="image" src="https://github.com/user-attachments/assets/0ffd2fe5-ba0c-403c-a65f-f73415beb7a6" />


