I built this by first evaluating the data files one by one and checking the time and date columns, then tried to calculate the naive baseline. Once I got into that I saw that some shifts are night shifts, going from 22:00 to 06:00 for example.
To counter this I put in a rule that if the clock-out time is earlier than the clock-in time, it assumes the shift crossed midnight and adds 24 hours to the calculation, so the duration comes out accurate instead of negative.

I then looked at what classifies as a breach, which is 10 hours over the standard 45-hour week (55 hours total). 
I used something called a sigmoid function to turn that gap into a risk score, so someone sitting at 55 hours projected shows as lower risk than someone heading toward 90, and we express it as a percentage out of 100.

I then realised this was inaccurate because the baseline assumes everyone works the same pace every week. 
This is actually where the naive baseline falls short: if you know how someone typically works, maybe a lot more hours on certain days than others, that history makes your prediction far more accurate than just looking at this week's pace so far. So instead of only using days worked, we took all of an employee's past weeks, calculated their average hours and the standard deviation of those hours, and used that to compare against and predict this week.

Also had to be careful not to let the model "see" the current week's outcome when training on it, because that creates data leakage and can produce false confidence in predictions that wouldn't hold up on real, unseen weeks.

This data was straightforward, maybe not enough info to have a fully accurate model, but accurate enough to beat the naive baseline. The data was interesting to work with, and for sorting and joining I just used standard functions for that. The actual XGBoost model was just normal parameter tuning, and I also wanted to weigh false negatives more heavily than false positives, since missing a real breach costs money while a false alarm just means a wasted check-in. Once that was tuned, the model was working as intended. I also added a test so that if the data changed or was removed mid-build it would flag me, and set that up as a notification rule in the actual Slack app.

Thanks again for opportunity i love solving such Problems and enjoyed this task alot.

https://www.loom.com/share/02ba801b2dd343d1a92125756ba7acc1 Website
