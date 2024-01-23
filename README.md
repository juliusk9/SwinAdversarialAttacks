The datasets are stored as follows:
1. In datasets, there lives a test and train folder.
2. Each of these folders contains, respectively, images the model is trained on and images used for testing the performance.
    A. Within the train folder there is:
        i. a folder original, with the images copied from the BreaKHis dataset
        ii. a folder pgd_attack, with two nested folders:
            a. a folder swin, which contains all perturbed images from the original generated by attacking the swin model, used for retraining the swin model
            b. a folder resnet, which contains all perturbed images from the original generated by attacking the resnet model, used for retraining the resnet model
    B. Within the test folder there is:
        i. a folder original, with the images copied from the BreaKHis dataset
        ii. the folders one_pixel_attack, pgd_attack and triangle_attack, which all contain:
            a. a folder swin with all test images perturbed by attacking the swin network 
            a. a folder resnet with all test images perturbed by attacking the resnet network 
