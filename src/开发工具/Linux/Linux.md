---
title: Linux命令
weight: 2
---
Linux命令记录
<!-- more -->
## 1.Linux配置免密登录
### 1.1 安装OpenSSH
### 1.2 在Windows生成密钥对
以`管理员身份`打开命令提示符`cmd`,输入`ssh-keygen`，一路`Enter`

### 1.3 修改SSH配置信息
`vim /etc/ssh/sshd_config`
配置如下
```shell
RSAAuthentication yes
PubkeyAuthentication yes
GSSAPIAuthentication no
GSSAPICleanupCredentials yes
PasswordAuthentication no
```

### 1.4 在Linux生成密钥对

`ssh-keygen`，也是一路`Enter`

生成的密钥对存放在当前目录下的`.ssh`目录下，`id_rsa`是`私钥`，`id_rsa.pub`是`公钥`

### 1.5 复制密钥

在`.ssh`目录输入`touch authorized_keys`,创建`authorized_keys`文件,在将`Windows`复制的公钥粘贴到`Linux`里,`cat /root/id_rsa.pub >>~/.ssh/authorized_keys`

设置权限

```shell
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### 1.6 重启ssh

`service sshd restart`

### 免密连接SSH

`ssh 用户名@Linux的IP地址`
