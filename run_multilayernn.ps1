python -u train.py `
	--model multilayernn `
	--hidden-dim 512 `
	--epochs 100 `
	--weight-decay 0.1 `
	--log-interval 50 `
	--lr 0.001 `
	--log-id 7 | tee multilayernn.log
