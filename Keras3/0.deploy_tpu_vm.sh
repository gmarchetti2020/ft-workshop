gcloud compute disks create your-hyperdisk \
    --size 1024  \
    --zone your-zone-name \
    --type hyperdisk-balanced

gcloud compute tpus tpu-vm create your-tpu-vm \
    --project your-project-id \
    --zone=your-zone-name \
    --accelerator-type=v6e-4 \
    --version=v2-alpha-tpuv6e \
    --data-disk source=projects/your-project-id/zones/your-zone-name/disks/your-hyperdisk,mode=read-write