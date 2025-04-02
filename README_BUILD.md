```
cd data
```
Create file lexicon.txt
```
python3 prepare.py
```

```
#### INSTALL MFA  ####
!wget https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh -qO ./miniconda.sh
# !wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -qO $PWD/miniconda.sh # for Apple M1
!bash ./miniconda.sh -b -p ./miniconda
conda create -n aligner -c conda-forge montreal-forced-aligner=2.2.15 -y --quiet


conda activate aligner


```

```
mfa train \
    --num_jobs $(nproc) \
    --use_mp \
    --clean \
    --overwrite \
    --no_textgrid_cleanup \
    --single_speaker \
    --output_format json \
    --output_directory dataset \
    dataset ./lexicon.txt vbx_mfa
```

```
conda deactivate
```

```
python3 train_ftdata.py
```
