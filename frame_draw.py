def frame_draw(data):
    import numpy as np
    clip_data = []
    metrics = {}
    for n, value in enumerate(data['people']):
        if n == 0:
            box_colour = (0,0,255)
            plot_colour = 'bo'
        else:
            box_colour = (255,0,0)
            plot_colour = 'ro'

        player = data['people'][n]['pose_keypoints_2d']
        while player.count(0):
            player.remove(0)
        player_x = []
        player_y = []
        for i in range(0, int(len(player)/3)):
            x = player.pop(-3)
            y = player.pop(-2)
            player.pop(-1)
            player_x.append(x)
            player_y.append(y)

        sum_x = np.sum(player_x)
        sum_y = np.sum(player_y)
        x_c = round(sum_x/len(player_x))
        y_c = round(sum_y/len(player_y))
        centroid = (x_c, y_c)
        width = round (max( abs( x_c-max(player_x) ), abs( x_c-min(player_x) ) ) * 2.2)
        height = round (max( abs( y_c-max(player_y) ), abs( y_c-min(player_y) ) ) * 2.2)
        # Due to the centroid often being located slightly lower than the hips,
        # the box is not exactly centered about the centroid and instead shifted up 
        # slightly to avoid cutting off the head 
        box_origin = ( round(x_c-width/1.9), round(y_c-height/1.9) )

        metrics['{0}'.format(n)] = [box_origin, width, height, centroid,
                box_colour, (player_x, player_y)]
    clip_data.append(metrics)
    return clip_data
