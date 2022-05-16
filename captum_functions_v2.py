import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import LayerGradCam
from captum.attr import LayerAttribution
from captum.attr import visualization as viz
from captum.attr import GuidedGradCam

if torch.cuda.is_available():     # Make sure GPU is available
    device = torch.device("cuda:0")
    kwar = {'num_workers': 8, 'pin_memory': True}
    cpu = torch.device("cpu")
else:
    print("Warning: CUDA not found, CPU only.")
    device = torch.device("cpu")
    kwar = {}
    cpu = torch.device("cpu")

def plot_saliency_map(model, img_tensor, pred_label_idx):
    saliency = Saliency(model)
    img_tensor.requires_grad = True
    grads = saliency.attribute(img_tensor, target=pred_label_idx)
    
    _ = viz.visualize_image_attr_multiple(np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(img_tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          methods=["original_image", "blended_heat_map"], signs=["all","absolute_value"],
                                          show_colorbar=True, titles=["Original Image","Overlayed Gradient Magnitudes"], fig_size=(16,12))

                                          
def plot_ig(model, img_tensor, pred_label_idx):
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(img_tensor, target=pred_label_idx, n_steps=200, internal_batch_size=100)
    
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(img_tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          methods=['original_image','heat_map'],
                                          cmap=None,
                                          show_colorbar=True,
                                          signs=['all','absolute_value'],
                                          titles=['Original Image', 'Integrated Gradients'],
                                          outlier_perc=1,
                                          fig_size=(16,12))
                                          
def plot_ig_smooth(model, img_tensor, pred_label_idx):
    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(img_tensor, n_samples=10, nt_type='smoothgrad_sq', target=pred_label_idx, internal_batch_size=100)
    
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(img_tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          methods=['original_image','heat_map'],
                                          cmap=None,
                                          show_colorbar=True,
                                          signs=['all','absolute_value'],
                                          titles=['Original Image', 'Smooth Integrated Gradients'],
                                          fig_size=(16,12))
                                          
def plot_grad_shap(model, img_tensor, pred_label_idx):
    gradient_shap = GradientShap(model)
    # Defining baseline distribution of images
    rand_img_dist = torch.cat([img_tensor * 0, img_tensor * 1])
    
    attributions_gs = gradient_shap.attribute(img_tensor,
                                              n_samples=50,
                                              stdevs=0.0001,
                                              baselines=rand_img_dist,
                                              target=pred_label_idx)
    
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(img_tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          methods=['original_image','heat_map'],
                                          cmap=None,
                                          show_colorbar=True,
                                          signs=['all','absolute_value'],
                                          titles=['Original Image', 'Gradient Shapley'],
                                          outlier_perc=1,
                                          fig_size=(16,12))
                                          
def get_single_shap_plt(model, img_tensor, pred_label_idx, img_tensor_unnormalized, fig_ax):
    gradient_shap = GradientShap(model)
    rand_img_dist = torch.cat([img_tensor * 0, img_tensor * 1])
    
    attributions_gs = gradient_shap.attribute(img_tensor,
                                              n_samples=50,
                                              stdevs=0.0001,
                                              baselines=rand_img_dist,
                                              target=pred_label_idx)
    
    fig, ax = fig_ax
    
    vis_fig,vis_ax = viz.visualize_image_attr(np.transpose(attributions_gs.squeeze().detach().numpy(), (1,2,0)),
                                              np.transpose(img_tensor_unnormalized.squeeze().detach().numpy(), (1,2,0)),
                                              "blended_heat_map",
                                              "absolute_value",
                                              cmap=None,
                                              show_colorbar=False,
                                              alpha_overlay=0.65,
                                              plt_fig_axis=(fig,ax), outlier_perc=5, use_pyplot=False)
    return vis_fig, vis_ax

def get_single_occ_plt(model, img_tensor, pred_label_idx, img_tensor_unnormalized, fig_ax):
    occlusion = Occlusion(model)
    
    
    attributions_occ = occlusion.attribute(img_tensor,
                                           strides = (3, 8, 8),
                                           target=pred_label_idx,
                                           sliding_window_shapes=(3,15,15),
                                           baselines=0)
    
    fig, ax = fig_ax
    
    vis_fig,vis_ax = viz.visualize_image_attr(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(img_tensor_unnormalized.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              "blended_heat_map",
                                              "absolute_value",
                                              cmap=None,
                                              show_colorbar=False,
                                              alpha_overlay=0.5,
                                              plt_fig_axis=(fig,ax), outlier_perc=5, use_pyplot=False)
    return vis_fig, vis_ax  

def get_single_ig_plt(model, img_tensor, pred_label_idx, img_tensor_unnormalized, fig_ax):
    integrated_gradients = IntegratedGradients(model)
    
    
    attributions_ig = integrated_gradients.attribute(img_tensor,
                                                     target=pred_label_idx,
                                                     n_steps=10)
    
    fig, ax = fig_ax
    
    vis_fig,vis_ax = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(img_tensor_unnormalized.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              "blended_heat_map",
                                              "absolute_value",
                                              cmap=None,
                                              show_colorbar=False,
                                              alpha_overlay=0.5,
                                              plt_fig_axis=(fig,ax), outlier_perc=5, use_pyplot=False)
    return vis_fig, vis_ax 

def get_single_gradcam_plt(model, img_tensor, pred_label_idx, img_tensor_unnormalized, fig_ax):
    layer_gc = LayerGradCam(model, model.layer4[-1]) # For ResNet
    
    
    attributions_gc = layer_gc.attribute(img_tensor,target=pred_label_idx)
    
    upsampled_attr = LayerAttribution.interpolate(attributions_gc, (224, 224))
    
    fig, ax = fig_ax
    
    vis_fig,vis_ax = viz.visualize_image_attr(np.transpose(upsampled_attr.squeeze().repeat(3,1,1).cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(img_tensor_unnormalized.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              "blended_heat_map",
                                              "absolute_value",
                                              cmap=None,
                                              show_colorbar=False,
                                              alpha_overlay=0.5,
                                              plt_fig_axis=(fig,ax), use_pyplot=False)
    return vis_fig, vis_ax

def get_single_guidedgradcam_plt(model, img_tensor, pred_label_idx, img_tensor_unnormalized, fig_ax):
    guided_gc = GuidedGradCam(model, model.layer4[-1]) # For ResNet
    
    
    attributions_ggc = guided_gc.attribute(img_tensor,target=pred_label_idx)
    
    #upsampled_attr = LayerAttribution.interpolate(attributions_ggc, (224, 224))
    
    fig, ax = fig_ax
    
    vis_fig,vis_ax = viz.visualize_image_attr(np.transpose(attributions_ggc.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              np.transpose(img_tensor_unnormalized.squeeze().cpu().detach().numpy(), (1,2,0)),
                                              "blended_heat_map",
                                              "absolute_value",
                                              cmap=None,
                                              show_colorbar=False,
                                              alpha_overlay=0.5,
                                              plt_fig_axis=(fig,ax), use_pyplot=False)
    return vis_fig, vis_ax

def plot_occlusion(model, img_tensor, pred_label_idx, strides = (3,8,8), sliding_window_shapes=(3,15, 15)):
    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(img_tensor,
                                           strides = strides,
                                           target=pred_label_idx,
                                           sliding_window_shapes=sliding_window_shapes,
                                           baselines=0)
    
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(img_tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          methods=['original_image','heat_map'],
                                          cmap=None,
                                          show_colorbar=True,
                                          signs=['all','positive'],
                                          titles=['Original Image', 'Occlusions'],
                                          outlier_perc=2,
                                          fig_size=(16,12))
                                          
def plot_deep_lift(model, img_tensor, pred_label_idx):
    dl = DeepLift(model)
    attr_dl = dl.attribute(img_tensor, baselines=0)
    
    _ = viz.visualize_image_attr_multiple(np.transpose(attr_dl.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(img_tensor.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          methods=['original_image','blended_heat_map'],
                                          cmap=None,
                                          show_colorbar=True,
                                          signs=['all','all'],
                                          titles=['Original Image', 'Deep Lift'],
                                          fig_size=(16,12))

    
# Expected Gradients, Implementierung nach Verena Schweinstetter
def compute_ex_gradients(model, images, target_class_idx, cuda=False):
    # do the pre-processing
    gradients = []
    for image in images:
        # preprocess the image for the model
        # image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        image = image.to(device).requires_grad_(True)
        if cuda:
            image = torch.tensor(image, 
                                 dtype=torch.float32, 
                                 device = torch.device('cuda:0'), 
                                 requires_grad = True)
        else:
            image = torch.tensor(image, 
                                 dtype=torch.float32, 
                                 device = torch.device('cpu'), 
                                 requires_grad = True)
        # use the model
        output = model(image)
        output = F.softmax(output, dim=1)
        # if target_class_idx is None:
        #     target_class_idx = torch.argmax(output, 1).item()
        index = np.ones((output.size()[0], 1)) * target_class_idx
        index = torch.tensor(index, dtype=torch.int64)
        if cuda:
            index = index.cuda()
        output = output.gather(1, index)
        # clear grad
        model.zero_grad()
        output.backward()
        gradient = image.grad.detach().cpu().numpy()[0]
        # postprocessing
        gradient = gradient.transpose(1, 2, 0)
        gradients.append(gradient)
    #gradients = np.array(gradients)
    return gradients #, target_class_idx

def expected_gradients(model,
                       baselines,
                       image,
                       target_class_idx):
    
    expected_gradients = np.zeros(shape = [224, 224, 3])
    
    # how many baselines?
    m_steps = len(baselines)
    
    # 1. Generate random uniformly distributed alphas.
    alphas = torch.rand(m_steps)
    
    for i in range(0, m_steps):
        
        # 2. Generate interpolated inputs between baseline and input.
        baseline_x = baselines[i]
        delta = image - baseline_x
        interpolated_img = baseline_x - alphas[i]*delta

        # 3. Compute gradients between model outputs and interpolated inouts.
        gradient = compute_ex_gradients(model = model,
                                  images=interpolated_img,
                                  target_class_idx=target_class_idx, 
                                  cuda = False)

        # 4. Multiply x - x'
        delta = torch.squeeze(delta).permute(1,2,0).numpy()
        gradient *= delta
    
        # 5. Add up
        expected_gradients += np.squeeze(gradient)
      
    # 6. Divide and transform to torch tensor
    expected_gradients /= m_steps
    expected_gradients = torch.unsqueeze(torch.from_numpy(expected_gradients),0)
    return expected_gradients

def plot_img_exp_attributions(model,
                              baselines,
                              img_tensor,
                              orig_image,
                              target_class_idx,
                              fig_ax,
                              overlay_alpha=0.4):
    attributions = expected_gradients(model = model,
                                      baselines=baselines,
                                      image=img_tensor,
                                      target_class_idx=target_class_idx)
    
    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = torch.sum(torch.abs(attributions), axis=-1).permute(1, 2, 0)
    orig_image = torch.squeeze(orig_image).permute(1,2,0)
    #image = torch.squeeze(image).permute(1,2,0)
    fig, ax = fig_ax
      
    ax.imshow(attribution_mask)
    ax.imshow(orig_image, alpha=overlay_alpha)
    ax.axis('off')
    
    return fig, ax