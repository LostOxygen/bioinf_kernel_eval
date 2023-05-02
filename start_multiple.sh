#!/bin/bash
trap "kill 0" EXIT

help()
{
    printf "Usage: start_multiple [ -b | --batch_size ] [ -e | --epochs ] [ -l | --learning_rate]"
    exit 2
}

while [[ "$#" -gt 0 ]];
do
	case "$1" in
		-b | --batch_size )
			BATCH_SIZE="$2"
			shift
			;;
		-l | --learning_rate )
			LEARNING_RATE="$2"
			shift
			;;
		-a | --epochs )
			EPOCHS="$2"
			shift
			;;
		-* | --* )
			shift
			break
			;;
		* )
			printf "Unexpected argument: $1"
			help
			;;
	esac
	shift
done

{
	LOADED_GPUS="$(ls /proc/driver/nvidia/gpus/ | wc -l)";
} || {
	LOADED_GPUS=0;
}

if [[ "$LOADED_GPUS" -eq 0 ]]; then
	DEVICE=-1
fi

printf "\n##########################################################"
printf "\n## Found $LOADED_GPUS GPU(s) on this machine"
printf "\n##########################################################\n"

python main.py --model_type vgg11 --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --epochs "$EPOCHS" --gpu 0 && python main.py --model_type vgg11 --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --epochs "$EPOCHS" --gpu 0 --depthwise && python main.py --model_type resnet34 --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --epochs "$EPOCHS" --gpu 0 &
python main.py --model_type vgg13 --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --epochs "$EPOCHS" --gpu 1 && python main.py --model_type vgg13 --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --epochs "$EPOCHS" --gpu 1 --depthwise && python main.py --model_type resnet34 --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --epochs "$EPOCHS" --gpu 1 --depthwise &
python main.py --model_type vgg16 --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --epochs "$EPOCHS" --gpu 2 && python main.py --model_type vgg16 --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --epochs "$EPOCHS" --gpu 2 --depthwise &
python main.py --model_type vgg19 --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --epochs "$EPOCHS" --gpu 3 && python main.py --model_type vgg19 --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --epochs "$EPOCHS" --gpu 3 --depthwise &

wait
