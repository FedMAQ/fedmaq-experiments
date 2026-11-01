# ntfy notifications

`notify_run.sh` defaults to the public `ntfy.sh` service. An unreserved topic is
readable and writable by anyone who knows its name; the name is the access
capability. The bearer token used by the runner does not make an unreserved topic
private.

Keep the topic and token in the ignored local `.env` file or environment variables.
Use a fresh random topic name, keep real values out of tracked files, and never
print the topic in help, errors, test output, logs, or examples. `.env.copy` must use
obvious placeholders. Pass the token to `curl` through a short-lived config file
created with owner-only permissions; keep it out of process arguments and remove
the config immediately after each request.

Notifications may contain only the event status and elapsed duration. Keep host
names, commands, paths, standard output/error, and failure details local. If a
future workflow needs private notification content, use an access-controlled
reserved topic or a private server instead of adding detail to an unreserved topic.

The phone subscribes to the same topic name configured in the local `.env`. When
changing the topic, update the local value and phone subscription together. A
connectivity test should confirm delivery without printing the topic name.
