import argparse
import os
import json
import numpy as np
import tensorflow.compat.v1 as tf
import open3d
import time

from models import pointnet_seg
import importlib
from utils.metric import ConfusionMatrix

from data.tum_mls_dataset import TUMMLSDataset


class Predictor:
    def __init__(self, checkpoint_path, num_classes, hyper_params):
        # Get ops from graph
        with tf.device("/gpu:0"):
            # Placeholder
            MODEL = importlib.import_module("pointnet_seg")  # import network module
            pl_points, labels_pl = MODEL.placeholder_inputs(64, hyper_params["num_point"])
            pl_is_training = tf.placeholder(tf.bool, shape=())
            print("pl_points shape", tf.shape(pl_points))

            # Prediction
            pred, end_points = MODEL.get_model(pl_points, pl_is_training)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            # Saver
            saver = tf.train.Saver()

            # Graph for interpolating labels
            # Assuming batch_size == 1 for simplicity
            pl_sparse_points = tf.placeholder(tf.float32, (None, 3))
            pl_sparse_labels = tf.placeholder(tf.int32, (None,))
            #pl_dense_points = tf.placeholder(tf.float32, (None, 3))
            #pl_knn = tf.placeholder(tf.int32, ())
            # dense_labels, dense_colors = interpolate_label_with_color(
            #     pl_sparse_points, pl_sparse_labels, pl_dense_points, pl_knn
            # )

        self.ops = {
            "pl_points": pl_points,
            "pl_is_training": pl_is_training,
            "pred": pred,
            "pl_sparse_points": pl_sparse_points,
            "pl_sparse_labels": pl_sparse_labels,
            'loss': loss
            # "pl_dense_points": pl_dense_points,
            # "pl_knn": pl_knn,
            # "dense_labels": dense_labels,
            # "dense_colors": dense_colors,
        }

        # Restore checkpoint to session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        saver.restore(self.sess, checkpoint_path)
        print("Model restored")

    def predict(self, batch_data, run_metadata=None, run_options=None):
        """
        Args:
            batch_data: batch_size * num_point * 6(3)

        Returns:
            pred_labels: batch_size * num_point * 1
        """
        is_training = False
        feed_dict = {
            self.ops["pl_points"]: batch_data,
            self.ops["pl_is_training"]: is_training,
        }

        
        if run_metadata is None:
            run_metadata = tf.RunMetadata()
        if run_options is None:
            run_options = tf.RunOptions()

        pred_val = self.sess.run(
            [self.ops["pred"]],
            options=run_options,
            run_metadata=run_metadata,
            feed_dict=feed_dict,
        )
        pred_val = pred_val[0]  # batch_size * num_point * 1
        pred_labels = np.argmax(pred_val, 2)  # batch_size * num_point * 1
        return pred_labels


if __name__ == "__main__":
    np.random.seed(0)

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="# samples, each contains num_point points_centered",
    )
    parser.add_argument("--ckpt", default='log/semantic/model.ckpt', help="Checkpoint file")
    parser.add_argument("--set", default="validation", help="train, validation, test")
    flags = parser.parse_args()
    hyper_params = json.loads(open("semantic_no_color.json").read())

    # Create output dir
    output_dir = os.path.join("log/seg_result", "sparse")
    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    dataset = TUMMLSDataset(
        num_points_per_sample=hyper_params["num_point"],
        split=flags.set,
        box_size_x=hyper_params["box_size_x"],
        box_size_y=hyper_params["box_size_y"],
        use_color=hyper_params["use_color"],
        path=hyper_params["data_path"],
    )
    batch_size = 64

    
    print(dataset.get_total_num_points()/hyper_params["num_point"])
    print(dataset.get_num_batches(batch_size))
    flags.num_samples = dataset.get_total_num_points()/hyper_params["num_point"]
    print(int(np.ceil(flags.num_samples / batch_size)))
    
    # Model
    
    predictor = Predictor(
        checkpoint_path=flags.ckpt,
        num_classes=dataset.num_classes,
        hyper_params=hyper_params,
    )

    # Process each file
    cm = ConfusionMatrix(9)

    for semantic_file_data in dataset.list_file_data:
        print("Processing {}".format(semantic_file_data))

        # Predict for num_samples times
        points_collector = []
        pd_labels_collector = []

        # If flags.num_samples < batch_size, will predict one batch
        for batch_index in range(dataset.get_num_batches(batch_size)):
            current_batch_size = min(
                batch_size, flags.num_samples - batch_index * batch_size
            )   # 每一轮batch的数量

            # Get data
            points_centered, points, gt_labels, colors = semantic_file_data.sample_batch(
                batch_size=current_batch_size,
                num_points_per_sample=hyper_params["num_point"],
            )

            # (bs, 8192, 3) concat (bs, 8192, 3) -> (bs, 8192, 6)
            if hyper_params["use_color"]:
                points_centered_with_colors = np.concatenate(
                    (points_centered, colors), axis=-1
                )
            else:
                points_centered_with_colors = points_centered

            # Predict
            s = time.time()
            pd_labels = predictor.predict(points_centered_with_colors)
            print(
                "Batch size: {}, time: {}".format(current_batch_size, time.time() - s)
            )

            # Save to collector for file output
            points_collector.extend(points)
            pd_labels_collector.extend(pd_labels)

            # Increment confusion matrix
            cm.increment_from_list(gt_labels.flatten(), pd_labels.flatten())

        # Save sparse point cloud and predicted labels
        file_prefix = os.path.basename(semantic_file_data.file_path_without_ext)

        sparse_points = np.array(points_collector).reshape((-1, 3))
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(sparse_points)
        pcd_path = os.path.join(output_dir, file_prefix + ".pcd")
        open3d.io.write_point_cloud(pcd_path, pcd)
        print("Exported sparse pcd to {}".format(pcd_path))

        sparse_labels = np.array(pd_labels_collector).astype(int).flatten()
        pd_labels_path = os.path.join(output_dir, file_prefix + ".labels")
        np.savetxt(pd_labels_path, sparse_labels, fmt="%d")
        print("Exported sparse labels to {}".format(pd_labels_path))

    cm.print_metrics()
