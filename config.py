"""
Define the hyper parameters here.
"""

import os


class config:
    # global parameters
    cuda_device = "cuda:0"
    base_path = os.path.abspath(os.path.dirname(__file__))  # path to this project
    caption_model = 'att2all'  # 'show_tell', 'att2all', 'adaptive_att', 'spatial_att'
    # refer to README.md for more info about each model
    output_path = "data/output/DenseNet201_test/"
    dataset_type = 'coco'
    # dataset parameters
    dataset_image_path = os.path.join(base_path, '/home/dane/Images/coco2014/')
    dataset_caption_path = os.path.join(base_path, '/home/dane/Images/coco2014/karpathy/dataset_coco.json')
    dataset_output_path = os.path.join(base_path, output_path)  # folder with data files saved by preprocess.py
    dataset_basename = 'DenseNet201_test'  # any name you want

    # preprocess parameters
    captions_per_image = 5
    min_word_freq = 5  # words with frenquence lower than this value will be mapped to '<UNK>'
    max_caption_len = 50  # captions with length higher than this value will be ignored,
    # with length lower than this value will be padded from right side to fit this length

    # word embeddings parameters
    embed_pretrain = True  # false: initialize embedding weights randomly
    # true: load pre-trained word embeddings
    embed_path = os.path.join(base_path,
                              '/home/dane/Images/glove/glove.6B.300d.txt')  # only makes sense when `embed_pretrain = True`
    embed_dim = 512  # dimension of word embeddings
    # only makes sense when `embed_pretrain = False`
    fine_tune_embeddings = False  # fine-tune word embeddings?

    # model parameters
    attention_dim = 128  # dimension of attention network
    # you only need to set this when the model includes an attention network
    decoder_dim = 128  # dimension of decoder's hidden layer
    dropout = 0.5
    model_path = os.path.join(base_path, output_path, 'checkpoints/')  # path to save checkpoints
    model_basename = 'ResnetTest'  # any name you want

    # training parameters
    validation_measure = 'cider'  # bleu_4 cider
    epochs = 2
    batch_size = 500
    pretrained_encoder = 'DenseNet201'  # DenseNet201 Resnet101 Regnet32
    fine_tune_encoder = False  # fine-tune encoder or not
    encoder_lr = 1e-4  # learning rate of encoder (if fine-tune)
    decoder_lr = 4e-4  # learning rate of decoder
    grad_clip = 5.  # gradient threshold in clip gradients
    # checkpoint = os.path.join(base_path, output_path,
    #                           'checkpoints/checkpoint_DenseNet201_glove300_decoder_dim_512_attention_dim_512_ft_embeddings_false_fine_tune_encoder_false.pth.tar')  # path to load checkpoint, None if none
    checkpoint = None
    workers = 0  # num_workers in dataloader
    tau = 1.  # penalty term τ for doubly stochastic attention in paper: show, attend and tell
    # you only need to set this when 'caption_model' is set to 'att2all'
    # tensorboard
    tensorboard = True  # enable tensorboard or not?
    log_dir = os.path.join(base_path, output_path, 'logs/att2all/')  # folder for saving logs for tensorboard
    # only makes sense when `tensorboard = True`
