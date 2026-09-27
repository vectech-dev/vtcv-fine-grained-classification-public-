from vtcv_fine_grained_classification.src.visualization.visualize_att_map import create_and_save_subplot


class _VisConfig:
    max_pixel = 255
    min_pixel = 0


def save_visualizations(images, labels, out, batch_id, save_dir):
    maps = out["attention_map"]
    if maps.shape[1] < 2:
        return
    lo = maps.flatten(2).min(2)[0][..., None, None]
    hi = maps.flatten(2).max(2)[0][..., None, None]
    maps = (maps - lo) / (hi - lo + 1e-7)
    data = {
        "im": images,
        "labels": labels,
        "im_crop": out.get("images_crop", images),
        "im_drop": out.get("images_drop", images),
        "att_maps": maps,
        "prop": out["prop"],
        "topk": out["topk"],
    }
    create_and_save_subplot(data, _VisConfig, batch_id, save_dir)
