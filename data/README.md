# Access to raw data

In this README file you will see how to access and process the data until the generation of the models.

## Download the dataset

You can do it in two ways:

### 1. Via kaggle hub in your code

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("muhammadatefelkaffas/eeg-muse2-motor-imagery-brain-electrical-activity")

print("Path to dataset files:", path)

```

### 2. Use the download link on Kaggle website

[Link to the dataset](https://www.kaggle.com/datasets/muhammadatefelkaffas/eeg-muse2-motor-imagery-brain-electrical-activity)

## About the Dataset (copied from Kaggle website)

This dataset was collected by me as a means for training machine / deep learning models in an EEG motor imagery classification. This was one of my roles as a machine learning engineer for the graduation project in Faculty of Artificial Intelligence, KafrElsheikh university.
I am profoundly grateful to all the technical support and advice provided by the project's supervisor, Dr. Mona AlNaggar

The goal is to perform motor imagery classification (left, right, relaxed) and translate JUST thoughts into action.

I used Muse2 Headband with 4 electrodes and Muse Monitor android app to start the recording sessions, and export the CSVs.

This high dimensional, temporal-data was collected in both subject-independent and subject-dependent contexts with the help of 19 healthy subjects (12 males, 7 females) in different states aged between 19 and 68 as a means of training for various deterministic and non-deterministic machine learning models to carry out Motor imagery classification task. 20 columns, where we have 5 powerbands (Alpha, beta, theta, delta, gamma) per each of the 4 sensor-electrodes, were of significance to the motor imagery classification. I didn't use the raw data. However, raw data is exported via muse monitor too so you can use it as more insights can be extracted out of the raw data and thus use 4 columns only (AF7, AF8, TP9, TP10).
Features like gyro, accelerometer weren't an area of interest for this EEG brain analysis or motor imagery classification. Feature engineering techniques like PCA, ICA can be beneficial especially for the raw data scenario.

Motor Imagery is one class of the event-related potentials. It is the imagination of motion, without doing any actual movement.

For the elements column, there are instances of blink, Marker 1, 2, 3.
An Important assumption to mention about the data markers is that :

Marker 1 -> left motor imagery

Marker 2 -> right motor imagery

Marker 3 -> Relaxed state (which is an intermediate phase between right and left in the conducted motor imagery experiments for 19 subjects)

Data was later split into training, testing and validation portions.
The experiment involves various steps, fitting the muse2 headband to the subject's head, the experiment conductor carefully looking over Mind Monitor android app and starts a recording session to collect the data, objects are present on the right and left of the experiment subject.
Usually, relaxing state is maintained at first then left imagery then relax then right imagery and so on.

Before heading onto the experiment, subjects were trained upon the official Muse app, meditative sessions so we are sure of their ability to maintain their focus especially in the intermediate step between left and right imagery.

Experiment setup in comfortable and stable lightened places :-
The volunteer begins by sitting down on a comfortable chair, their arms parallel and resting on a table. Prior to this, they are trained in meditation using the Muse app to help them achieve a relaxed state.
Two cups of water are placed on either side of the participant, each 5 cm away from a hand and within their line of sight. The volunteer then sits comfortably, raises their chin slightly, and keeps their head steady to avoid noise in the EEG data. Their eyes rotate to the left or right, looking over the cup without moving their head.

The experiment conductor then instructs the participant to engage in motor imagery, imagining picking up the cup on their left with their left hand and drinking, but without any actual hand movement. This state of focus is maintained for a duration of 0.5, 1, 2, or 3 minutes, depending on the subject’s attention span. The same process is repeated for the cup on the right with the right hand.

If the experiment conductor notices a decline in the concentration level, indicating a lower attention span, they ask the volunteer to concentrate more on the motor imagery and apply a visual stimulus to the cup. These steps are repeated several times, ranging from 2 to 3, to capture clean and accurate EEG data while the volunteer is engaged in motor imagery tasks. It’s a careful balance of physical stillness and mental activity.

Potential applications for the motor imagery classification: Translating thoughts into actions for controlling a game, helping the handicapped control their surroundings especially in a smart home environment.

Obviously, the performance metric for such motor imagery task would be the F1 score which is harmonic mean between precision and recall.

## Process the data and recreate the models

1. First you must download the dataset.
2. Put the data in the current folder named: `data/eeg_muse2_moto_imagery_brain_ electrical_activity`
3. Go to the `notebooks` folder and use `requirements.txt` to generate the proper python env.
4. Go to the `sripts` folder run the `preprocess.py` file that should generate the preprocessed data.
5. Run the notebook `final_models_creation.ipynb` to generates the models.

## Other files

You will see the `notebooks/manual_feature_extraction_fuzzyfication.ipynb` notebook where a first attempt of model creation as been done.
