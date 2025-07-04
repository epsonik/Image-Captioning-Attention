"""
Define the hyper parameters here.
"""

import os


class config:
    # global parameters
    cuda_device = "cuda:0"
    base_path = os.path.abspath(os.path.dirname(__file__))  # path to this project
    caption_model = 'spatial_att'  # 'show_tell', 'att2all', 'adaptive_att', 'spatial_att'
    # refer to README.md for more info about each model
    output_path = "data/output/spatial_50_Densenet201_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true/"
    dataset_type = 'coco'
    # dataset parameters
    dataset_image_path = os.path.join(base_path, '/mnt/dysk2/dane/coco2014/')
    dataset_caption_path = os.path.join(base_path, '/mnt/dysk2/dane/coco2014/karpathy/dataset_coco.json')
    dataset_output_path = os.path.join(base_path, output_path)  # folder with data files saved by preprocess.py
    dataset_basename = 'spatial_50_Densenet201_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true'  # any name you want

    # preprocess parameters
    captions_per_image = 5
    min_word_freq = 5  # words with frenquence lower than this value will be mapped to '<UNK>'
    max_caption_len = 50  # captions with length higher than this value will be ignored,
    # with length lower than this value will be padded from right side to fit this length

    # word embeddings parameters
    embed_pretrain = True  # false: initialize embedding weights randomly
    # true: load pre-trained word embeddings
    embed_path = os.path.join(base_path,
                              '/mnt/dysk2/dane/glove/glove.6B.300d.txt')  # only makes sense when `embed_pretrain = True`
    # embed_path = os.path.join(base_path,
    #                           '/mnt/dysk2/dane/fastText/wiki-news-300d-1M-subword.vec')  # only makes sense when `embed_pretrain = True`
    embed_dim = 300  # dimension of word embeddings
    # only makes sense when `embed_pretrain = False`
    fine_tune_embeddings = True  # fine-tune word embeddings?

    # model parameters
    attention_dim = 128  # dimension of attention network
    # you only need to set this when the model includes an attention network
    decoder_dim = 512  # dimension of decoder's hidden layer
    dropout = 0.5
    model_path = os.path.join(base_path, output_path, 'checkpoints/')  # path to save checkpoints
    model_basename = 'spatial_50_Densenet201_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true'  # any name you want

    # training parameters
    epochs = 50
    batch_size = 10
    pretrained_encoder = 'DenseNet201'  # DenseNet201 Resnet101 Regnet32 DenseNet121 DenseNet161 Resnet152
    fine_tune_encoder = True  # fine-tune encoder or not
    encoder_lr = 1e-4  # learning rate of encoder (if fine-tune)
    decoder_lr = 4e-4  # learning rate of decoder
    grad_clip = 5.  # gradient threshold in clip gradients
    # checkpoint = os.path.join(base_path, output_path,
    #                           'checkpoints/checkpoint_adaptive_DenseNet201_decoder_dim_512_fine_tune_encoder_true_fine_tune_embeddings_true_fastText-epoch-22.pth.tar')  # path to load checkpoint, None if none
    checkpoint = None
    workers = 0  # num_workers in dataloader
    tau = 1.  # penalty term τ for doubly stochastic attention in paper: show, attend and tell
    # you only need to set this when 'caption_model' is set to 'att2all'
    # tensorboard
    tensorboard = True  # enable tensorboard or not?
    log_dir = os.path.join(base_path, output_path, 'logs/att2all/')  # folder for saving logs for tensorboard
    # only makes sense when `tensorboard = True`
