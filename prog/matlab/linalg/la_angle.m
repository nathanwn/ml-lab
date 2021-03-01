function omega = la_angle(u, v)
    % omega = angle between two vectors $u$ and $v$
    cos_omega = dot(u, v) / (norm(u) * norm(v));
    omega = acos(cos_omega);
end