scoring = {
    'pass_yd': .04,
    'pass_td': 4,
    'pass_2pt': 2,
    'pass_int': -1,
    'rush_yd': .1,
    'rush_td': 6,
    'rush_2pt': 2,
    'rec_rcpt': .5,
    'rec_yd': .1,
    'rec_td': 6,
    'rec_2pt': 2
}
df['pass_yd'] * .04
df['pass_td'] * 4
df['pass_2pt'] * 2
df['pass_int'] * -1

df['rush_yd'] * .1
df['rush_td'] * 6
df['rush_2pt'] * 2

df['rec_rcpt'] * .5
df['rec_yd'] * .1
df['rec_td'] * 6
df['rec_2pt'] * 2

df['fg_1_39_made'] * 3
df['fg_40_49_made'] * 4
df['fg_50_made'] * 5
df['pat_made'] * 1
df['pat_missed'] * -1
df['xp_made'] * 1

df['def_td'] * 6
(df['def_pts_allowed'] == 0) * 10
(df['def_pts_allowed'] >= 1 and df['def_pts_allowed'] <= 6) * 7
(df['def_pts_allowed'] >= 7 and df['def_pts_allowed'] <= 13) * 4
(df['def_pts_allowed'] >= 14 and df['def_pts_allowed'] <= 20) * 2
(df['def_pts_allowed'] >= 21 and df['def_pts_allowed'] <= 27) * 1
(df['def_pts_allowed'] >= 35) * -4
df['def_sck'] * 1
df['def_int'] * 2
df['def_fum_rec'] * 2
df['def_fum_forced'] * 1
df['def_sfty'] * 5
df['def_fum_forced'] * 1
df['def_kick_blocked'] * 2

df['fumble'] * -1
df['fumbles_lost'] * -2
df['fumble_rec_td'] * 6
