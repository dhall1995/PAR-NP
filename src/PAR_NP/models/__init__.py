## NOTE: Most of this code is taken/refactored from the original neural processes repo
## All credit goes to the original authors (https://github.com/wesselb/neuralprocesses)
import neuralprocesses as nps


def patch_model(d, patches):
    """Patch a loaded model.

    Args:
        d (dict): Output of :func:`torch.load`.

    Returns:
        dict: `d`, but patched.
    """
    with out.Section("Patching loaded model"):
        # Loop over patches.
        for patch in patches:
            base_from, base_to = patch.split(":")

            # Try to apply the patch.
            applied_patch = False
            for k in list(d["weights"].keys()):
                if k.startswith(base_from):
                    applied_patch = True
                    tail = k[len(base_from) :]
                    d["weights"][base_to + tail] = d["weights"][k]
                    del d["weights"][k]

            # Report whether the patch was applied.
            if applied_patch:
                out.out(f'Applied patch "{patch}".')
            else:
                out.out(f'Did not apply patch "{patch}".')
    return d


def construct_model(model_name, **model_kwargs):
    # Construct the model.
    if model_name == "cnp":
        model = nps.construct_gnp(
            dim_x=model_kwargs["dim_x"],
            dim_yc=(1,) * model_kwargs["dim_y"],
            dim_yt=model_kwargs["dim_y"],
            dim_embedding=model_kwargs["dim_embedding"],
            enc_same=model_kwargs["enc_same"],
            num_dec_layers=model_kwargs["num_layers"],
            width=model_kwargs["width"],
            likelihood="het",
            transform=model_kwargs["transform"],
        )
    elif model_name == "gnp":
        model = nps.construct_gnp(
            dim_x=model_kwargs["dim_x"],
            dim_yc=(1,) * model_kwargs["dim_y"],
            dim_yt=model_kwargs["dim_y"],
            dim_embedding=model_kwargs["dim_embedding"],
            enc_same=model_kwargs["enc_same"],
            num_dec_layers=model_kwargs["num_layers"],
            width=model_kwargs["width"],
            likelihood="lowrank",
            num_basis_functions=model_kwargs["num_basis_functions"],
            transform=model_kwargs["transform"],
        )
    elif model_name == "np":
        model = nps.construct_gnp(
            dim_x=model_kwargs["dim_x"],
            dim_yc=(1,) * model_kwargs["dim_y"],
            dim_yt=model_kwargs["dim_y"],
            dim_embedding=model_kwargs["dim_embedding"],
            enc_same=model_kwargs["enc_same"],
            num_dec_layers=model_kwargs["num_layers"],
            width=model_kwargs["width"],
            likelihood="het",
            dim_lv=model_kwargs["dim_embedding"],
            transform=model_kwargs["transform"],
        )
    elif model_name == "acnp":
        model = nps.construct_agnp(
            dim_x=model_kwargs["dim_x"],
            dim_yc=(1,) * model_kwargs["dim_y"],
            dim_yt=model_kwargs["dim_y"],
            dim_embedding=model_kwargs["dim_embedding"],
            enc_same=model_kwargs["enc_same"],
            num_heads=model_kwargs["num_heads"],
            num_dec_layers=model_kwargs["num_layers"],
            width=model_kwargs["width"],
            likelihood="het",
            transform=model_kwargs["transform"],
        )
    elif model_name == "agnp":
        model = nps.construct_agnp(
            dim_x=model_kwargs["dim_x"],
            dim_yc=(1,) * model_kwargs["dim_y"],
            dim_yt=model_kwargs["dim_y"],
            dim_embedding=model_kwargs["dim_embedding"],
            enc_same=model_kwargs["enc_same"],
            num_heads=model_kwargs["num_heads"],
            num_dec_layers=model_kwargs["num_layers"],
            width=model_kwargs["width"],
            likelihood="lowrank",
            num_basis_functions=model_kwargs["num_basis_functions"],
            transform=model_kwargs["transform"],
        )
    elif model_name == "anp":
        model = nps.construct_agnp(
            dim_x=model_kwargs["dim_x"],
            dim_yc=(1,) * model_kwargs["dim_y"],
            dim_yt=model_kwargs["dim_y"],
            dim_embedding=model_kwargs["dim_embedding"],
            enc_same=model_kwargs["enc_same"],
            num_heads=model_kwargs["num_heads"],
            num_dec_layers=model_kwargs["num_layers"],
            width=model_kwargs["width"],
            likelihood="het",
            dim_lv=model_kwargs["dim_embedding"],
            transform=model_kwargs["transform"],
        )
    elif model_name == "convcnp":
        model = nps.construct_convgnp(
            points_per_unit=model_kwargs["points_per_unit"],
            dim_x=model_kwargs["dim_x"],
            dim_yc=(1,) * model_kwargs["dim_y"],
            dim_yt=model_kwargs["dim_y"],
            likelihood="het",
            conv_arch=args.arch,
            unet_channels=model_kwargs["unet_channels"],
            unet_strides=model_kwargs["unet_strides"],
            conv_channels=model_kwargs["conv_channels"],
            conv_layers=model_kwargs["num_layers"],
            conv_receptive_field=model_kwargs["conv_receptive_field"],
            margin=model_kwargs["margin"],
            encoder_scales=model_kwargs["encoder_scales"],
            transform=model_kwargs["transform"],
        )
    elif model_name == "convgnp":
        model = nps.construct_convgnp(
            points_per_unit=model_kwargs["points_per_unit"],
            dim_x=model_kwargs["dim_x"],
            dim_yc=(1,) * model_kwargs["dim_y"],
            dim_yt=model_kwargs["dim_y"],
            likelihood="lowrank",
            conv_arch=args.arch,
            unet_channels=model_kwargs["unet_channels"],
            unet_strides=model_kwargs["unet_strides"],
            conv_channels=model_kwargs["conv_channels"],
            conv_layers=model_kwargs["num_layers"],
            conv_receptive_field=model_kwargs["conv_receptive_field"],
            num_basis_functions=model_kwargs["num_basis_functions"],
            margin=model_kwargs["margin"],
            encoder_scales=model_kwargs["encoder_scales"],
            transform=model_kwargs["transform"],
        )
    elif model_name == "convnp":
        if model_kwargs["dim_x"] == 2:
            # Reduce the number of channels in the conv. architectures by a factor
            # $\sqrt(2)$. This keeps the runtime in check and reduces the parameters
            # of the ConvNP to the number of parameters of the ConvCNP.
            model_kwargs["unet_channels"] = tuple(
                int(c / 2**0.5) for c in model_kwargs["unet_channels"]
            )
            model_kwargs["dws_channels"] = int(model_kwargs["dws_channels"] / 2**0.5)
        model = nps.construct_convgnp(
            points_per_unit=model_kwargs["points_per_unit"],
            dim_x=model_kwargs["dim_x"],
            dim_yc=(1,) * model_kwargs["dim_y"],
            dim_yt=model_kwargs["dim_y"],
            likelihood="het",
            conv_arch=args.arch,
            unet_channels=model_kwargs["unet_channels"],
            unet_strides=model_kwargs["unet_strides"],
            conv_channels=model_kwargs["conv_channels"],
            conv_layers=model_kwargs["num_layers"],
            conv_receptive_field=model_kwargs["conv_receptive_field"],
            dim_lv=16,
            margin=model_kwargs["margin"],
            encoder_scales=model_kwargs["encoder_scales"],
            transform=model_kwargs["transform"],
        )
    elif model_name == "fullconvgnp":
        model = nps.construct_fullconvgnp(
            points_per_unit=model_kwargs["points_per_unit"],
            dim_x=model_kwargs["dim_x"],
            dim_yc=(1,) * model_kwargs["dim_y"],
            dim_yt=model_kwargs["dim_y"],
            conv_arch=args.arch,
            unet_channels=model_kwargs["unet_channels"],
            unet_strides=model_kwargs["unet_strides"],
            conv_channels=model_kwargs["conv_channels"],
            conv_layers=model_kwargs["num_layers"],
            conv_receptive_field=model_kwargs["conv_receptive_field"],
            kernel_factor=model_kwargs["fullconvgnp_kernel_factor"],
            margin=model_kwargs["margin"],
            encoder_scales=model_kwargs["encoder_scales"],
            transform=model_kwargs["transform"],
        )
    else:
        raise ValueError(f'Invalid model "{model_name}".')

    return model
