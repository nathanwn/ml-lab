function omega = la_angle_deg(u, v)
    %omega = angle, in degree, between two vectors $u$ and $v$
    omega = rad2deg(la_angle(u, v));
end