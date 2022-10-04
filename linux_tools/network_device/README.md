To show all network devices
```
ip link
```
example output is like `lo, enp8s0, docker0`. lo means loop back.

And to analyze network interface state
```
ethtools -s ecat0
```
It shows something like
```
NIC statistics:
     rx_packets: 347640758
     tx_packets: 347729711
     rx_bytes: 297801768874
     tx_bytes: 297902152848
     rx_broadcast: 347640758
     tx_broadcast: 347729690
     rx_multicast: 0
     tx_multicast: 21
     rx_errors: 152415
     tx_errors: 0
     tx_dropped: 0
     multicast: 0
     collisions: 0
     rx_length_errors: 0
     rx_over_errors: 0
     rx_crc_errors: 76220
     rx_frame_errors: 76195
     rx_no_buffer_count: 0
     rx_missed_errors: 0
     tx_aborted_errors: 0
     tx_carrier_errors: 0
     tx_fifo_errors: 0
     tx_heartbeat_errors: 0
     tx_window_errors: 0
     tx_abort_late_coll: 0
     tx_deferred_ok: 0
     tx_single_coll_ok: 0
     tx_multi_coll_ok: 0
     tx_timeout_count: 0
     tx_restart_queue: 0
     rx_long_length_errors: 0
     rx_short_length_errors: 0
     rx_align_errors: 76195
     tx_tcp_seg_good: 0
     tx_tcp_seg_failed: 0
     rx_flow_control_xon: 0
     rx_flow_control_xoff: 0
     tx_flow_control_xon: 0
     tx_flow_control_xoff: 0
     rx_csum_offload_good: 0
     rx_csum_offload_errors: 0
     rx_header_split: 0
     alloc_rx_buff_failed: 0
     tx_smbus: 0
     rx_smbus: 0
     dropped_smbus: 0
     rx_dma_failed: 0
     tx_dma_failed: 0
     rx_hwtstamp_cleared: 0
     uncorr_ecc_errors: 0
     corr_ecc_errors: 0
     tx_hwtstamp_timeouts: 0
```
If value of `rx_crc_errors` and `rx_frame_errors` large, it indicates that something is wrong with ether device.
