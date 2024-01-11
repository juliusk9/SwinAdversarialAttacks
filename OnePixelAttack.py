import os
import torch
from utils import save_image, make_dirs

from scipy.optimize import differential_evolution


# Maybe function not needed because we have our own perturb image function.
def perturb_image_one_pixel(x, img):
    img = torch.clone(img)

    y_pos, x_pos, *rgb = x

    rgb_tensor = torch.tensor(rgb, dtype=img.dtype)

    img[0, :, int(y_pos), int(x_pos)] = rgb_tensor

    return img
 


def probability_classes(perturbation, image, org_class, model, device):
    # For debugging purposes
    #print("Perturbation in prob_clas", perturbation)
    #print("Image at pert pixels", image[0, :, int(perturbation[0]), int(perturbation[1])])

    # Perturb image and store in pert_image
    pert_image = perturb_image_one_pixel(perturbation, torch.clone(image))

    #print("Org Image after perturbation", image[0, :, int(perturbation[0]), int(perturbation[1])])
    #print("Perturbed image", pert_image[0, :, int(perturbation[0]), int(perturbation[1])])

    #print("Dimensions of ")
    #print(np.shape(pert_image))
    #print(np.shape(image))

    comparison = torch.eq(image[0], pert_image[0])

    if len(torch.nonzero(~comparison)) == 0:
        print("No difference between image and perturbed image")


    #print("number of pixels not equal:", [x for x in torch.eq(image, pert_image) if not x])
    
    output = model(torch.Tensor(pert_image).to(device))

    probabilities = torch.nn.functional.softmax(output, dim=1)

    adjusted_probability = probabilities[0][org_class]

    #print("Adj prob:", adjusted_probability)

    return adjusted_probability.to('cpu').detach()


def predict(perturbation, image, org_class, model, device, image_list, minimize=True):

    # print(len(perturbation))
    # print(perturbation)

#     image_list.append(image)


#     if len(image_list) > 1:
#         # print("Checking original images")
#         for i in range(1, len(image_list)):
#             if len(torch.nonzero(~torch.eq(image_list[i][0], image_list[i-1][0]))) > 0:
#                 # print("Source images are not the same")
#                 # print(torch.nonzero(~torch.eq(image_list[i][0], image_list[i-1][0])))
#         image_list.pop(0)

    #print("OriginalImage", image)

    attacked_image = perturb_image_one_pixel(perturbation, image)

    comparison = torch.eq(image[0], attacked_image[0])

    if len(torch.nonzero(~comparison)) == 0:
        print("No difference between image and perturbed image")


    #print("Attacked_image", attacked_image)
    #print("Tensored Attackedimage", torch.Tensor(attacked_image))

    original_output = model(image.to(device))
    perturbed_output = model(attacked_image.to(device))

#     print("Original", original_output[0][0])
#     print("Perturbed", perturbed_output[0][0])

    return perturbed_output[0][org_class].item()


def attack_success(perturbation, image, org_class, model, device):

    attacked_image = perturb_image_one_pixel(perturbation, torch.clone(image))

    # print("Checking if succesful")
    confidences = model(attacked_image.to(device))[0]
    predicted_class = torch.argmax(confidences)

    # print(confidences)
    # print(predicted_class, org_class)

    if predicted_class != org_class:
        return True


def attack(index, model, device, image, label, pixel_count=1, maxiter=50, popsize=10):
    make_dirs("advanced_one_pixel")
    # # Returns image from the test dataset
    # sample = dataset[index]

    # # Image information
    # org_img = sample[0].unsqueeze(0)
    org_class = torch.argmax(label).item()

    #print(torch.eq(image, image))

    # Clean up memory
    torch.cuda.empty_cache()
 
    model.to(device)
    
    bounds = [[(0, 256), (0,256), (0,1), (0,1), (0,1)] * pixel_count]
    # The population has size popsize * (N - N_equal)
    popsize = popsize // len(bounds[0])

    # This is the function taken by the differential evolution of SciPy to find the minimum of.
    image_list = []
    def func(perturb):
        #print("\n Called as func")
        return predict(perturb, torch.clone(image), org_class, model, device, image_list)
    
    # Function that keeps track of the best solution found so far.
    def callback(x, convergence=None):
        # print("\n Called as callback")
        return attack_success(x, torch.clone(image), org_class, model, device)

    result = differential_evolution(
        # removed from below: recombination=1, atol=-1, 
        func, bounds[0], maxiter=maxiter, popsize=popsize, recombination=1, atol=-1, 
        callback=callback, polish=False
    )

    print("\n Result", result.message)
    print("Success", result.success)

    model.to('cpu')

    pert_image = perturb_image_one_pixel(result.x, torch.clone(image))

    org_output = model(image)
    pert_output = model(torch.Tensor(pert_image))
    
    print(org_output, pert_output)
    
    pert_pred = torch.nn.functional.softmax(model(torch.Tensor(pert_image)), dim=1)
    pert_class = torch.argmax(pert_pred).detach().numpy()

    print(org_class, pert_class)
    
    # checken of de predicted class met originele plaatje gelijk aan de true class (uit label), zo niet is ie verkeerd geclassificeerd en betekend dat niet dat de pertubation successful is.
    successful = pert_class != org_class

    if successful:
        save_image(pert_image.squeeze(), os.path.join(os.getcwd(), "dataset", "advanced_one_pixel", str(index) + "-attacked.png"))
        save_image(image.squeeze(), os.path.join(os.getcwd(), "dataset", "advanced_one_pixel", str(index) + "-original.png"))

    return successful