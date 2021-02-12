def s_range(start, stop, step):
	if step == -1:
		step = stop
	r = set(range(start, stop, step))
	r.add(stop)
	return r
