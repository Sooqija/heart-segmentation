from monai.networks.nets import UNETR

def unetr():
    model = UNETR(
        in_channels=1,
        out_channels=8,
        img_size=(128, 128, 128),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )
    
    return model