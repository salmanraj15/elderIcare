# elderIcare Dataset Specification



**Document status:** Draft

**Dataset version:** 0.1

**Project:** elderIcare

**Last updated:** 2026-08-28



---



## 1. Purpose



The elderIcare dataset is a collection of original audio recordings created specifically for research and development of the elderIcare acoustic event detection system.



The initial objective is to train a machine-learning model to classify short audio segments into predefined acoustic-event categories.



The dataset is intended for experimental research and development. It is not intended to diagnose medical conditions or independently determine whether an emergency has occurred.



---



## 2. Data Ownership and Provenance



The initial elderIcare dataset will consist of recordings created by the project team.



No third-party audio datasets are required for the initial training dataset.



Each recording should have documented provenance, including:



* Recording identifier

* Recording date

* Recording environment

* Recording device

* Sampling rate

* Audio format

* Intended class

* Relevant recording notes



Recordings should only be included when the project has the appropriate rights and consent to use them.



---



## 3. Initial Classes



The first dataset version contains six acoustic classes.



| Class        | Description                                                             |

| ------------ | ----------------------------------------------------------------------- |

| `background` | Normal environmental or room sound without a target event               |

| `speech`     | Ordinary human speech                                                   |

| `help_call`  | A deliberate verbal call intended to attract attention or request help  |

| `impact`     | A significant impact, crash, knock, or collision-like sound             |

| `cough`      | One or more audible cough events                                        |

| `alarm`      | An audible alarm signal such as a smoke, CO, security, or similar alarm |



---



## 4. Class Definitions



### 4.1 `background`



Represents normal environmental sound.



Examples:



* Quiet room

* Fan

* Air conditioning

* Distant household noise

* Normal movement

* Low-level room ambience



A recording should be labelled `background` when none of the other target events is clearly present.



---



### 4.2 `speech`



Represents ordinary human speech that is not specifically a request for assistance.



Examples:



* Conversation

* Reading aloud

* Asking an ordinary question

* Talking on the phone

* Television speech when clearly captured as speech



A normal spoken sentence should not be labelled `help_call` merely because it contains words associated with assistance.



---



### 4.3 `help_call`



Represents an intentional verbal attempt to attract attention or request assistance.



Examples:



* Calling someone's name loudly to attract attention

* Calling for help

* Repeatedly asking for assistance

* A clearly intentional distress/help vocalization



The classification should be based on the acoustic event and recording context defined for the dataset.



The model should not be considered capable of determining whether a person is actually in danger solely from this class.



---



### 4.4 `impact`



Represents a significant sudden impact or collision-like acoustic event.



Examples:



* Object hitting the floor

* Chair striking the floor

* Strong knock

* Object falling

* Significant collision



An impact recording does **not** mean that a person has fallen.



For safety reasons, `impact` should remain an acoustic classification. Any later interpretation such as possible fall detection must be handled by a separate system layer.



---



### 4.5 `cough`



Represents audible coughing.



Examples:



* Single cough

* Repeated cough

* Short coughing episode



The dataset should record different natural variations where possible.



The presence of coughing must not be interpreted as a medical diagnosis.



---



### 4.6 `alarm`



Represents an audible alarm signal.



Examples:



* Smoke alarm

* Carbon-monoxide alarm

* Security alarm

* Other clearly identifiable warning alarm



Different alarm patterns should be represented where possible.



---



## 5. Ambiguous Recordings



Some recordings may contain multiple classes.



For example:



```text

speech → impact → speech

```



or:



```text

background → cough → background

```



For the initial dataset, recordings should preferably be constructed or selected so that the target event is reasonably isolated.



Recordings containing multiple significant events should be documented explicitly.



Ambiguous samples should not be forced into a class merely to increase dataset size.



A future dataset version may introduce multi-label or temporal event annotation.



---



## 6. Recording Requirements



Initial recordings should preferably use:



* WAV format

* Uncompressed PCM audio

* A consistent sampling rate

* A consistent channel configuration

* Consistent microphone positioning where appropriate



The exact technical recording specification will be finalized before dataset collection begins.



---



## 7. Dataset Splitting



The dataset will eventually be divided into:



* Training set

* Validation set

* Test set



The test set must remain isolated from training decisions.



Where recordings are made by the same person or in the same environment, care must be taken to prevent highly similar recordings from appearing across training and test sets.



This is necessary to reduce the risk of measuring memorization rather than generalization.



---



## 8. Dataset Metadata



Each recording should eventually have associated metadata.



Example:



```text

recording_id: REC_000001

class: background

date: YYYY-MM-DD

device: <recording device>

sampling_rate: <Hz>

channels: 1

duration_seconds: <duration>

environment: living_room

notes: quiet room with low background fan noise

```



Metadata should not contain unnecessary personally identifying information.



---



## 9. Privacy



Audio recordings can contain sensitive information.



The project should follow a privacy-by-design approach.



Where practical:



* Record only necessary audio.

* Avoid recording private conversations unnecessarily.

* Obtain appropriate consent from people who are recorded.

* Store recordings securely.

* Do not publish private recordings.

* Remove unnecessary identifying information from metadata.

* Keep raw recordings separate from source code.



---



## 10. Dataset Quality



Dataset quality is more important than simply increasing the number of recordings.



We should aim for variation in:



* Speakers

* Speaking styles

* Distances from microphone

* Room acoustics

* Background noise

* Recording levels

* Event intensity

* Event duration



However, these variations should be introduced deliberately and documented.



---



## 11. Dataset Versioning



Dataset versions will follow a simple versioning scheme:



```text

v0.1

v0.2

v0.3

...

v1.0

```



Changes between versions should be documented.



A dataset version should not silently replace an earlier version.



---



## 12. Limitations



The initial dataset will have important limitations.



Potential limitations include:



* Limited number of speakers

* Limited environments

* Limited microphone types

* Artificially produced events

* Class imbalance

* Background noise differences

* Lack of real-world elder-care environments



Model performance must therefore be interpreted in the context of the dataset used for training and evaluation.



---



## 13. Future Work



Potential future dataset improvements include:



* More speakers

* More environments

* More microphone positions

* More realistic background noise

* Temporal event annotations

* Multi-label recordings

* Hard-negative examples

* Real-world deployment recordings collected with appropriate consent

* Evaluation specifically designed for Raspberry Pi deployment



---



## 14. Safety Boundary



elderIcare is an experimental acoustic-event detection system.



The system must not make unsupported claims such as:



* "A person has fallen."

* "A medical emergency is occurring."

* "The person requires medical treatment."



An acoustic event can provide evidence that something may have happened, but acoustic classification alone does not establish the underlying real-world event.



Any future alerting system should therefore communicate uncertainty and, where appropriate, require human verification.



