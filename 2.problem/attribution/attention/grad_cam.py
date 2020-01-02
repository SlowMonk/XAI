from gradcam.utils import  *

from imports import *


image_size = 32

class GradCAM(object):
    """Calculate GradCAM salinecy map.
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """

        b, c, h, w = input.size()
        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):
	"""Calculate GradCAM++ salinecy map.
    A simple example:
        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

	def __init__(self, model_dict, verbose=False):
		super(GradCAMpp, self).__init__(model_dict, verbose)

	def forward(self, input, class_idx=None, retain_graph=False):
		"""
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
		b, c, h, w = input.size()

		logit = self.model_arch(input)
		if class_idx is None:
			score = logit[:, logit.max(1)[-1]].squeeze()
		else:
			score = logit[:, class_idx].squeeze()

		self.model_arch.zero_grad()
		score.backward(retain_graph=retain_graph)
		gradients = self.gradients['value']  # dS/dA
		activations = self.activations['value']  # A
		b, k, u, v = gradients.size()

		alpha_num = gradients.pow(2)
		alpha_denom = gradients.pow(2).mul(2) + \
					  activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
		alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

		alpha = alpha_num.div(alpha_denom + 1e-7)
		positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
		weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

		saliency_map = (weights * activations).sum(1, keepdim=True)
		saliency_map = F.relu(saliency_map)
		saliency_map = F.upsample(saliency_map, size=(image_size, image_size), mode='bilinear', align_corners=False)
		saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
		saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

		return saliency_map, logit

	def adjust_image(ratio, trainloader, saliency_maps, eval_method):
		# set threshold
		print('adjust Image')
		data = trainloader.dataset.data
		img_size = data.shape[1:]  # mnist : (28,28), cifar10 : (32,32,3)
		nb_pixel = np.prod(img_size)
		threshold = int(nb_pixel * (1 - ratio))
		# rank indice
		re_sal_maps = saliency_maps.reshape(saliency_maps.shape[0], -1)
		indice = re_sal_maps.argsort().argsort()
		# get mask
		if eval_method == 'ROAR':
			mask = indice < threshold
		elif eval_method == 'KAR':
			mask = indice >= threshold
		mask = mask.reshape(data.shape)
		print(mask.shape)
		# remove
		trainloader.dataset.data = (data * mask).reshape(data.shape)

		return trainloader


def normalize(tensor, mean, std):
	if not tensor.ndimension() == 4:
		raise TypeError('tensor should be 4D')

	mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

	return tensor.sub(mean).div(std)


class Normalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, tensor):
		return self.do(tensor)

	def do(self, tensor):
		return normalize(tensor, self.mean, self.std)

	def undo(self, tensor):
		return denormalize(tensor, self.mean, self.std)

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def adjust_image(ratio, trainloader, saliency_maps, eval_method):
	# set threshold
	print('adjust Image')
	data = trainloader.dataset.data
	img_size = data.shape[1:]  # mnist : (28,28), cifar10 : (32,32,3)
	nb_pixel = np.prod(img_size)
	threshold = int(nb_pixel * (1 - ratio))
	# rank indice
	re_sal_maps = saliency_maps.reshape(saliency_maps.shape[0], -1)
	indice = re_sal_maps.argsort().argsort()
	# get mask
	if eval_method == 'ROAR':
		mask = indice < threshold
	elif eval_method == 'KAR':
		mask = indice >= threshold
	mask = mask.reshape(data.shape)
	print(mask.shape)
	# remove
	trainloader.dataset.data = (data * mask).reshape(data.shape)

	return trainloader
def find_resnet_layer(arch, target_layer_name):
	"""Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
	if 'layer' in target_layer_name:
		hierarchy = target_layer_name.split('_')
		layer_num = int(hierarchy[0].lstrip('layer'))
		if layer_num == 1:
			target_layer = arch.layer1
		elif layer_num == 2:
			target_layer = arch.layer2
		elif layer_num == 3:
			target_layer = arch.layer3
		elif layer_num == 4:
			target_layer = arch.layer4
		else:
			raise ValueError('unknown layer : {}'.format(target_layer_name))

		if len(hierarchy) >= 2:
			bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
			target_layer = target_layer[bottleneck_num]

		if len(hierarchy) >= 3:
			target_layer = target_layer._modules[hierarchy[2]]

		if len(hierarchy) == 4:
			target_layer = target_layer._modules[hierarchy[3]]

	else:
		target_layer = arch._modules[target_layer_name]

	return target_layer


def visualize_cam(mask, img):
	"""Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """

	mask = mask.cpu()
	heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)

	heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
	b, g, r = heatmap.split(1)
	heatmap = torch.cat([r, g, b])

	result = heatmap + img.cpu()
	result = result.div(result.max()).squeeze()

	return heatmap, result

def normalize_image(torch_img):
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.upsample(torch_img, size=(image_size, image_size), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)
    return torch_img,normed_torch_img

def get_camdic(net,type,layer):

	cam_dict = dict()

	resnet_model_dict = dict(type= type, arch=net, layer_name=layer, input_size=(28, 28))
	resnet_gradcam = GradCAM(resnet_model_dict, True)
	resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
	cam_dict[type] = [resnet_gradcam, resnet_gradcampp]
	return cam_dict

def make_masks(trainloader,cam_dict):
	masks=[]
	print('start->', masks)
	for idx, (img, target) in enumerate(trainloader):
		datapath = '/home/jake/Gits/AI college/XAI/2.problem/datab2/gradcam_masks.h5py'
		#if os.path.exists(datapath):

		#else:
		#	with h5py.File(datapath, 'w') as hf:
	#			hf.create_dataset('saliencies', data=masks)

		torch_img, normed_torch_img = normalize_image(img.cuda())

		for gradcam, gradcam_pp in cam_dict.values():
			mask, _ = gradcam(normed_torch_img)
			# print('mask->',mask.shape)
			# mask = cv2.resize(mask[0], (224, 224))
			mask = mask.squeeze()
			# print('mask->',mask)
			heatmap, result = visualize_cam(mask, torch_img)
			# print('heatpmap->',heatmap.shape,'result->',result.shape)
			# images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, result], 0))
			img = img.squeeze()
			img = (img).permute(1, 2, 0)
			#print(mask.shape)
			#print('appending',np.array(masks).shape)
			#masks.append(np.array(result.detach().cpu()))
			#print(heatmap.shape)
			masks.append(np.array(heatmap.detach().cpu()))
		#if idx==10: return masks
		if idx % 10000 == 0: print(idx, '/', trainloader.dataset.data.shape[0], '%')
		#if idx==15: break
	return masks

def generate_gradcam(trainloader,net,type,layer):
	images = []

	for i, (img, targets) in enumerate(trainloader):
		# print('오리지널:',img.shape)
		# img = img.squeeze()
		torch_img, normed_torch_img = normalize_image(img.cuda())
		cam_dict = get_camdic(net,type,layer)
		for gradcam, gradcam_pp in cam_dict.values():
			mask, _ = gradcam(normed_torch_img)
			heatmap, result = visualize_cam(mask, torch_img)

			images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, result], 0))
			img = img.squeeze()
			img = (img).permute(1, 2, 0)

		if i == 10: break
	images = make_grid(torch.cat(images, 0), nrow=3)
	img_name = 'temp2.jpg'
	output_dir = 'outputs'
	os.makedirs(output_dir, exist_ok=True)
	output_name = img_name
	output_path = os.path.join(output_dir, output_name)

	save_image(images, output_path)
	PIL.Image.open(output_path)