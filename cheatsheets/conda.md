# Conda Cheatsheet

## Delete/Remove an environment

```sh
$ conda env remove --name <envname>
```

## Install packages from `requirements.txt`

```sh
$ conda install --file requirements.txt
```

## Export environment to a yaml file

```sh
$ conda env export --name ENVNAME > envname.yml
```
