export PROJECT_ID=diesel-patrol-382622
export ACCELERATOR_TYPE=v6e-8
export ZONE=asia-northeast1-b
export REGION=asia-northeast1
export RUNTIME_VERSION=v2-alpha-tpuv6e
export QUEUED_RESOURCE_ID=briankang-$ACCELERATOR_TYPE-$ZONE
export TPU_NAME=$QUEUED_RESOURCE_ID


gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE

gcloud alpha compute tpus queued-resources list --project ${PROJECT_ID} --zone ${ZONE}

gcloud alpha compute tpus queued-resources create ${TPU_NAME} \
--node-id ${TPU_NAME} \
--project ${PROJECT_ID} \
--zone ${ZONE} \
--accelerator-type ${ACCELERATOR_TYPE} \
--runtime-version ${RUNTIME_VERSION} 

gcloud alpha compute tpus queued-resources delete ${QUEUED_RESOURCE_ID} --project ${PROJECT_ID} --zone ${ZONE} --force

# Delete TPU queued resources that are suspended
for i in $(gcloud alpha compute tpus queued-resources list --zone $ZONE --project $PROJECT_ID --filter="state=SUSPENDED" | awk '{print $1}'| tail -n +2)
do
  gcloud alpha compute tpus queued-resources delete $i --zone $ZONE --project $PROJECT_ID --force --quiet
done

# Delete TPU queued resources that are FAILED
for i in $(gcloud alpha compute tpus queued-resources list --zone $ZONE --project $PROJECT_ID --filter="state=FAILED" | awk '{print $1}'| tail -n +2)
do
  gcloud alpha compute tpus queued-resources delete $i --zone $ZONE --project $PROJECT_ID --force --quiet
done

##################
#### Start test
##################
gcloud alpha compute tpus queued-resources ssh ${TPU_NAME} --project ${PROJECT_ID} --zone ${ZONE}

export VENVNAME=tunixtext
virtualenv $VENVNAME
source $VENVNAME/bin/activate

# reactivate
export VENVNAME=tunixtext
source $VENVNAME/bin/activate

pip install --upgrade pip

pip install -q kagglehub

pip install -q tensorflow
pip install -q tensorboardX
pip install -q grain
pip install -q git+https://github.com/google/tunix
pip install -q git+https://github.com/google/qwix

pip uninstall -q -y flax
pip install -q git+https://github.com/google/flax.git

pip install -q datasets

# additional requirements
pip install tensorflow_datasets
pip install typing-extensions --upgrade
pip install typing --upgrade
pip install tpu-info

mkdir tunix
sudo chmod +777 tunix
cd tunix

# comment out the "Self" import from this line:
# https://github.com/google/tunix/blob/b7fcd9a391fea0c5e82fba161eabb177655005b9/tunix/models/gemma/gemma.py#L20
# and add this line of code
# from typing_extensions import Self
nano tunix/models/gemma/gemma.py

# call fine tuning
python3 tunix_train.py

# pip freeze output
absl-py==2.3.1
aiohappyeyeballs==2.6.1
aiohttp==3.12.14
aiosignal==1.4.0
array_record==0.7.2
astunparse==1.6.3
async-timeout==5.0.1
attrs==25.3.0
certifi==2025.7.14
charset-normalizer==3.4.2
chex==0.1.89
cloudpickle==3.1.1
datasets==4.0.0
dill==0.3.8
dm-tree==0.1.9
docstring_parser==0.16
einops==0.8.1
etils==1.13.0
filelock==3.18.0
flatbuffers==25.2.10
flax @ git+https://github.com/google/flax.git@2e779f579223275fd9ac9b59af98919f7c8a4d48
frozenlist==1.7.0
fsspec==2025.3.0
gast==0.6.0
google-pasta==0.2.0
grain==0.2.11
grpcio==1.73.1
h5py==3.14.0
hf-xet==1.1.5
huggingface-hub==0.33.4
humanize==4.12.3
idna==3.10
immutabledict==4.2.1
importlib_resources==6.5.2
jax==0.6.2
jaxlib==0.6.2
jaxtyping==0.3.2
kagglehub==0.3.12
keras==3.10.0
libclang==18.1.1
libtpu==0.0.17
Markdown==3.8.2
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
ml_dtypes==0.5.1
more-itertools==10.7.0
msgpack==1.1.1
multidict==6.6.3
multiprocess==0.70.16
namex==0.1.0
nest-asyncio==1.6.0
numpy==2.1.3
opt_einsum==3.4.0
optax==0.2.5
optree==0.16.0
orbax-checkpoint==0.11.19
packaging==25.0
pandas==2.3.1
promise==2.3
propcache==0.3.2
protobuf==4.21.12
psutil==7.0.0
pyarrow==20.0.0
Pygments==2.19.2
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.2
qwix @ git+https://github.com/google/qwix@f597c6e20534eb622366931dd74166d108e5a986
requests==2.32.4
rich==14.0.0
scipy==1.15.3
sentencepiece==0.2.0
simple-parsing==0.1.7
simplejson==3.20.1
six==1.17.0
tensorboard==2.19.0
tensorboard-data-server==0.7.2
tensorboardX==2.6.4
tensorflow==2.19.0
tensorflow-datasets==4.9.9
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorstore==0.1.76
termcolor==3.1.0
toml==0.10.2
toolz==1.0.0
tpu-info==0.4.0
tqdm==4.67.1
treescope==0.1.9
tunix @ git+https://github.com/google/tunix@976284361b1b8f1ce348940bca368a90aec8741e
typing==3.7.4.3
typing_extensions==4.14.1
tzdata==2025.2
urllib3==2.5.0
wadler_lindig==0.1.7
Werkzeug==3.1.3
wrapt==1.17.2
xxhash==3.5.0
yarl==1.20.1
zipp==3.23.0





# kill active sessions
ps -C main -o pid=|sudo xargs kill -9
ps -C python3 -o pid=|sudo xargs kill -9

# find pid of any accelerators that are still in process
sudo lsof /dev/vfio/*