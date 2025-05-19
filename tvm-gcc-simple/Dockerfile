FROM ubuntu:22.04

# Принимаем аргументы для UID и GID
ARG USER_ID=1000
ARG GROUP_ID=1000

# Установка timezone неинтерактивно
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

# Установка основных пакетов
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    ninja-build \
    llvm-14-dev \
    clang-14 \
    libedit-dev \
    libxml2-dev \
    zlib1g-dev \
    libz-dev \
    libzstd-dev \
    wget \
    curl \
    rsync \
    vim \
    mc \
    sudo \
    locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Генерация и настройка локалей
RUN locale-gen en_US.UTF-8 ru_RU.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
    echo "export LANG=en_US.UTF-8" >> /etc/bash.bashrc && \
    echo "export LC_ALL=en_US.UTF-8" >> /etc/bash.bashrc

# Установка переменных окружения для локалей
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US:en

# Добавление символьной ссылки python -> python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Установка Python зависимостей
RUN pip3 install --upgrade pip && \
    pip3 install numpy scipy cloudpickle \
    packaging psutil tornado typing_extensions \
    ml_dtypes onnx onnxruntime "cython>=0.29" \
    "onnxscript>=0.1.0" sympy

# Создание рабочей директории
WORKDIR /workspace

# Переменные окружения для TVM
ENV PYTHONPATH=/workspace/tvm/python:/workspace/tvm/topi/python
ENV TVM_HOME=/data/tvm
ENV Host_BUILD=/data/tvm/build
ENV Dev_BUILD=/data/tvm/build-riscv
ENV ZSTD_LIB_DIR=/usr/lib/x86_64-linux-gnu
ENV TOOLROOT=/opt/riscv
ENV RISCV_CFLAGS="-march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -O3"
ENV JOBS=4

# Создание директорий для скриптов и данных
RUN mkdir -p /workspace/scripts /data

# Скачивание и установка RISC-V тулчейна Xuantie
RUN mkdir -p /tmp/toolchain && \
    cd /tmp/toolchain && \
    wget -q https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1698113812618/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.8.0-20231018.tar.gz && \
    mkdir -p /opt/riscv && \
    tar -xzf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.8.0-20231018.tar.gz -C /opt/riscv --strip-components=1 && \
    rm -rf /tmp/toolchain && \
    /opt/riscv/bin/riscv64-unknown-linux-gnu-gcc --version

# Создание пользователя и группы для соответствия хостовому UID/GID
RUN groupadd -g ${GROUP_ID} builder && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash builder && \
    echo "builder ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/builder && \
    chmod 0440 /etc/sudoers.d/builder && \
    chown -R builder:builder /data /workspace

# Настройка .bashrc для пользователя
# --- добавляем всё, что должно оказаться в ~/.bashrc ---
RUN cat <<'EOF' >> /home/builder/.bashrc
# цветная подсказка и алиасы
export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "
alias la='ls -A'
alias l='ls -CF'
alias ll='ls -alhFp'
alias hcg='cat ~/.bash_history | grep '
alias 7zip='7z a -t7z -mx=9 -mfb=273 -ms -md=31 -myx=9 -mtm=- -mmt -mmtf -md=1536m -mmf=bt3 -mmc=10000 -mpb=0 -mlc=0 '
alias cls='clear;clear'
alias gcrs='git clone --recurse-submodules '
alias gprs='git pull  --recurse-submodules '

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]; then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# ========================================
path() {
    local old=$IFS
    IFS=:
    printf '%s\n' $PATH
    IFS=$old
}
# ---------------------------------
libs() {
    echo "LD_LIBRARY_PATH contents:"
    local old=$IFS
    IFS=:
    printf '%s\n' $LD_LIBRARY_PATH
    IFS=$old
}
EOF

# --- права на файл ---
RUN chown builder:builder /home/builder/.bashrc


# Рабочая директория
WORKDIR /workspace/scripts

# Скрипт входа в контейнер
CMD ["/bin/bash"]
