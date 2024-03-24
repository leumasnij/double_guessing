-- Please enter your code here:-- Nicolas Alt, 2014-09-04
-- Command-and-measure script
-- Works with extended wsg_50 ROS package
-- Tests showed about 20Hz rate
printf("%s\n",cmd.interface());
cmd.register(0xB0); -- Measure only
cmd.register(0xB1); -- Position control
cmd.register(0xB2); -- Speed control
def_speed = 5;
is_speed = false;
is_position = false;

-- Get number of FMF fingers
nfin = 0;
for i = 0,1 do
    if finger.type(i) == "fmf" then
        nfin = nfin + 1;
    else
        break;
    end
end
printf("#FMF fingers: %d\n", nfin)

function hasbit(x, p)
  return x % (p + p) >= p       
end

function process()
    id, payload = cmd.read();
    -- ==== Measurements (1) ====
    busy = mc.busy()
    blocked = mc.blocked()
    pos = mc.position();
    -- printf("Gcc %f\n",id)
    
    -- Position control
    if id == 0xB1 then
        cmd_width = bton({payload[2],payload[3],payload[4],payload[5]});
        cmd_speed = bton({payload[6],payload[7],payload[8],payload[9]});
        -- printf("Got command %f, %f\n", cmd_width, cmd_speed);
              print("set_pos");
        is_position = true;is_speed = false;
        if busy then mc.stop(); end
        mc.move(cmd_width, math.abs(cmd_speed), 0)
    -- Velocity control
    elseif id == 0xB2 then
        -- do_speed = hasbit(payload[1], 0x02);
        cmd_speed = bton({payload[6],payload[7],payload[8],payload[9]});
        --print("set_speed");
        --is_speed = true;
        --def_speed = cmd_speed;
        --mc.speed(cmd_speed);
        if cmd_speed == 0 then
            print("set_speed");
            mc.stop();
        else
            print("set_speed");
            is_speed = true;is_position = false;
            def_speed = cmd_speed;
            mc.speed(cmd_speed);
        end
    end
    
    
       ---- TEST: Stop in position mode---
    
    if blocked and is_position and pos> cmd_width then
        print("stop-----");
        mc.stop();
        is_position = false;
    end
       
    -- ==== Actions ====
    -- Stop if in speed mode
    -- print(blocked, is_speed, pos);
    if blocked and is_speed and pos <= 50 and def_speed < 0 then
        print("stop 0");
        mc.stop(); is_speed = false;
    end
    if blocked and is_speed and pos >= 50 and def_speed > 0 then
        print("stop 1");
        mc.stop(); is_speed = false;
    end           
    
    -- ==== Get measurements ====
    state = gripper.state();
    busy = mc.busy();
    blocked = mc.blocked();
    pos = mc.position();
    speed = mc.speed();
    force = mc.aforce();
    
    force_l = 0; force_r = 0;
    if nfin >= 1 then force_l = finger.data(0) end
    if nfin >= 2 then force_r = finger.data(1) end
    
    if cmd.online() then
        -- Only the lowest byte of state is sent!
        cmd.send(id, etob(E_SUCCESS), state % 256,
            { ntob(pos), ntob(speed), ntob(force), ntob(force_l), ntob(force_r)});
    end
end

while true do
    if cmd.online() then
        process()
        if not pcall(process) then
            print("Error occured")
            sleep(50)
        end
    else
        sleep(50)
    end
end
