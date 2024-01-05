# hangする原因
`class _SubscriberImpl(_TopicImpl)`の `receive_callback`をみてみると, msgが到達するたびに, 登録されたcallbacksをiterateしていることがわかる. つまり, cb1, cb2があり, cb1の終了がcb2の終了に依存している場合, cb1はハングする. 今までcbごとに別のスレッドで動いていると勘違いしていたが, 同一スレッドで呼ばれているのでこうなってしまう. 

```
def receive_callback(self, msgs, connection):
    """
    Called by underlying connection transport for each new message received
    @param msgs: message data
    @type msgs: [L{Message}]
    """
    # save reference to avoid lock
    callbacks = self.callbacks
    for msg in msgs:
        if self.statistics_logger:
            self.statistics_logger.callback(msg, connection.callerid_pub, connection.stat_bytes)
        for cb, cb_args in callbacks:
            self._invoke_callback(msg, cb, cb_args)
```
