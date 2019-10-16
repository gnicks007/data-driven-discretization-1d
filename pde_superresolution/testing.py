from pde_superresolution import equations
from pde_superresolution import model
from pde_superresolution import training
import numpy as np
import tensorflow as tf
from absl import logging
from scipy.integrate import solve_bvp
import json
import os
import copy
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

print("inside testing.py")

def define_hparams():
    NUM_X_POINTS = 256
    myparams = training.create_hparams(
    equation='TBG',
    conservative=False,
    numerical_flux=False,
    equation_kwargs=json.dumps({
        'num_points': NUM_X_POINTS,
    }),
    resample_factor=2,
    # neural network parameters
    model_target='coefficients',
    num_layers=3,
    filter_size=32,
    kernel_size=5,
    nonlinearity='relu',
    polynomial_accuracy_order=1,
    polynomial_accuracy_scale=1.0,
    ensure_unbiased_coefficients=False,
    coefficient_grid_min_size=4,
    # training parameters
    base_batch_size=128,
    learning_rates=[1e-4],
    learning_stops=[20000],
    frac_training=0.8,
    eval_interval=250,
    noise_probability=0.0,
    noise_amplitude=0.0,
    noise_type='white',
    # loss parameters ### Understand these better
    ground_truth_order=-1,
    num_time_steps=0,
    error_floor_quantile=0.1,
    error_scale=[np.nan],  # set by set_data_dependent_hparams
    error_floor=[np.nan],  # set by set_data_dependent_hparams
    error_max=0.0,
    absolute_error_weight=1.0,
    relative_error_weight=0.0,
    space_derivatives_weight=1.0,
    time_derivative_weight=0.0,
    integrated_solution_weight=0.0,
    )
    return myparams


def run_test():
    tbg = equations.TBGEquation(128,2, period=2) # in the initialization I specify the solution grid
    grid = tbg.grid
    u_batched_np =  np.sin(grid.solution_x).reshape(1,-1)
    u_batched_tf = tf.convert_to_tensor(u_batched_np, dtype=tf.float32)
    print("solution/coarse grid: ", grid.solution_num_points)
    print("reference/fine grid: ", grid.reference_num_points)
    print("batched_tf: ", u_batched_tf.shape, "not np: ", u_batched_np.shape)

    tbg_params = define_hparams()

    # predict_coefficients returns: dimensions [batch, x, derivative, coefficient].
    predict_coeffs = model.predict_coefficients(u_batched_tf, tbg_params)

    # predict_space_derivatives returns: dimensions [batch, x, derivative]
    predict_space_derivs = model.predict_space_derivatives(u_batched_tf, tbg_params)

    # equation value
    _, tbg_eqn = equations.from_hparams(tbg_params)
    # apply_space_derivatives plugs in predicted derivatives into equation 
    # returns: dimensions [batch, x]. point-wise value of the equation
    predict_eqn_value = model.apply_space_derivatives(predict_space_derivs, u_batched_tf, tbg_eqn) # we use actual grid values for u not predicted
    model_error = tf.reduce_mean(tf.square(predict_eqn_value))
    loss = model_error

    # outputs
    print("=============")
    print("predict coeffs: ", predict_coeffs.shape)
    print("predict space derivs: ", predict_space_derivs.shape)
    print("predict eqn value: ", predict_eqn_value.shape)
    print("model eror: ", model_error)
    print("=============")


    # labels: finite differences computed at high resolution, then subsampled
    # baseline: finite differences computed at low resolution
    # look at models.make_dataset
    # use polynomials.reconstruct


    #dataset = model.make_dataset(u, tbg_params, dataset_type=model.Dataset.TRAINING)
    #tensors = dataset.make_one_shot_iterator().get_next()
    #print(tensors)

def train_network():
    
    """ Run Training
    Args:
        snapshots: np.ndarray with shape [examples, x] with high-resolution
        training data.
        checkpoint_dir: directory to which to save model checkpoints.
        hparams: hyperparameters for training, as created by create_hparams().
    """
    hparams = define_hparams()
    tbg = equations.TBGEquation(128,2, period=2) # in the initialization I specify the solution grid
    grid = tbg.grid
    u_batched_np =  np.sin(grid.solution_x).reshape(1,-1)
    u_batched_tf = tf.convert_to_tensor(u_batched_np, dtype=tf.float32)
    hparams = copy.deepcopy(hparams)
    
    global_step = tf.compat.v1.train.get_or_create_global_step()

    logging.info('Training with hyperparameters:\n%r', hparams)
    equation_type = equations.equation_type_from_hparams(hparams)
    _, tbg_eqn = equations.from_hparams(hparams)

    if len(hparams.learning_rates) > 1:
        learning_rate = tf.train.piecewise_constant(
            global_step, boundaries=hparams.learning_stops[:-1],
            values=hparams.learning_rates)
    else:
        (learning_rate,) = hparams.learning_rates

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate, beta2=0.99)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        predict_space_derivs = model.predict_space_derivatives(u_batched_tf, hparams)
        predict_eqn_value = model.apply_space_derivatives(predict_space_derivs, u_batched_tf, tbg_eqn) # we use actual grid values for u not predicted
        loss = tf.reduce_mean(tf.square(predict_eqn_value))
        train_step = optimizer.minimize(loss, global_step=global_step)

    with tf.compat.v1.Session() as sess:
        initial_step = sess.run(global_step)
        for step in range(initial_step, hparams.learning_stops[-1]):
            sess.run(train_step)

def get_data_hparams():
    hparams = define_hparams()
    tbg = equations.TBGEquation(128,2, period=2) # in the initialization I specify the solution grid
    grid = tbg.grid
    u_batched_np =  np.sin(grid.solution_x).reshape(1,-1)
    u_batched_tf = tf.convert_to_tensor(u_batched_np, dtype=tf.float32)
    return u_batched_tf, hparams

def setup_training(snapshots: tf.Tensor, hparams: tf.contrib.training.HParams):
    _, tbg_eqn = equations.from_hparams(hparams)
    predict_space_derivs = model.predict_space_derivatives(snapshots, hparams)
    predict_eqn_value = model.apply_space_derivatives(predict_space_derivs, snapshots, tbg_eqn) # we use actual grid values for u not predicted
    loss = tf.reduce_mean(tf.square(predict_eqn_value))
    global_step = tf.train.get_or_create_global_step()

    if len(hparams.learning_rates) > 1:
        learning_rate = tf.train.piecewise_constant(
            global_step, boundaries=hparams.learning_stops[:-1],
            values=hparams.learning_rates)
    else:
        (learning_rate,) = hparams.learning_rates

    optimizer = tf.train.AdamOptimizer(learning_rate, beta2=0.99)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss, global_step=global_step)

    return loss, train_step


def training_loop(snapshots: tf.Tensor,
                  checkpoint_dir: str,
                  hparams: tf.contrib.training.HParams,
                  master: str = ''):
  """Run training.
  Args:
    snapshots: np.ndarray with shape [examples, x] with high-resolution
      training data.
    checkpoint_dir: directory to which to save model checkpoints.
    hparams: hyperparameters for training, as created by create_hparams().
    master: string master to use for MonitoredTrainingSession.
  Returns:
  """
  hparams = copy.deepcopy(hparams)
  logging.info('Training with hyperparameters:\n%r', hparams)

  hparams_path = os.path.join(checkpoint_dir, 'hparams.pbtxt')
  with tf.gfile.GFile(hparams_path, 'w') as f:
    f.write(str(hparams.to_proto()))

  logging.info('Setting up training')
  _, train_step = setup_training(snapshots, hparams)

  global_step = tf.train.get_or_create_global_step()

  logging.info('Variables: %s', '\n'.join(map(str, tf.trainable_variables())))

  logged_metrics = []
  equation_type = equations.equation_type_from_hparams(hparams)

  with tf.train.MonitoredTrainingSession(
      master=master,
      checkpoint_dir=checkpoint_dir,
      save_checkpoint_secs=300,
      config=training._session_config(),
      hooks=[training.SaveAtEnd(training.checkpoint_dir_to_path(checkpoint_dir))]) as sess:

    train_writer = tf.compat.v1.summary.FileWriter(
        os.path.join(checkpoint_dir, 'train'), sess.graph, flush_secs=60)

    initial_step = sess.run(global_step)

    with train_writer:
      for step in range(initial_step, hparams.learning_stops[-1]):
        sess.run(train_step)

#run_test()
#train_network()

snapshots, hparams = get_data_hparams()
#loss, train_step = setup_training(snapshots, hparams)

training_loop(snapshots, './', hparams)
