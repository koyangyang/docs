---
title: 常用Dokcer的命令
weight: 2
---
## Docker的基本操作
`Docker`也是目前很火的技术，当我在学习期间，因为学的东西比较多，部署环境很是麻烦，这时候就想把需要的环境搭建在`Docker`上，而且`Docker`搭建很是轻松，比如你安装`Centos`时候只需要几行命令便可以拉取安装，比起传统虚拟机安装方便快速很多，而且可以克隆部署多个。
<!-- more -->
## Docker的安装
- Ubuntu
- `curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun`
- Debian
- `curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun`
- Centos
- `curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun`

## Docker加速镜像
1. 编辑
`vim /etc/docker/daemon.json`
2. 修改
`{"registry-mirrors":["https://reg-mirror.qiniu.com/"]}`
推荐修改为`https://hub-mirror.c.163.com/`
3. 重启
`sudo systemctl daemon-reload`
`sudo systemctl restart docker`


## Docker安装
### Ubuntu脚本
```shell
curl -fsSL https://test.docker.com -o test-docker.sh
sudo sh test-docker.sh
```
## MySql
### MySql8
`docker run -itd --name mysql8 -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 mysql:8.0.33`
### MySql5
`docker run -itd --name mysql5 -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 mysql:5.7.42`

## Etcd
`docker run --name etcd -d -p 2379:2379 -p 2380:2380 -e ALLOW_NONE_AUTHENTICATION=yes bitnami/etcd:3.3.11 etcd`